import argparse
import sys
import os
import numpy as np
import nibabel as nib
# Assuming nrrd_utils is in data_io, and cli_utils for general helpers
try:
    from diffusemri.data_io.nrrd_utils import read_nrrd_data, write_nrrd_data
    from diffusemri.data_io.mhd_utils import read_mhd_data, write_mhd_data
    from diffusemri.data_io.analyze_utils import read_analyze_data, write_analyze_data # Added Analyze utils
    from diffusemri.data_io.cli_utils import save_nifti_data
except ImportError:
    # Fallback for direct script execution if diffusemri is not in PYTHONPATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from data_io.nrrd_utils import read_nrrd_data, write_nrrd_data
    from data_io.mhd_utils import read_mhd_data, write_mhd_data
    from data_io.analyze_utils import read_analyze_data, write_analyze_data # Added Analyze utils
    from data_io.cli_utils import save_nifti_data


def setup_nrrd_to_nifti_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--input_nrrd', required=True, help="Path to the input NRRD file.")
    parser.add_argument('--output_nifti', required=True, help="Path to save the output NIfTI file.")
    parser.add_argument('--output_bval', help="Path to save the output b-values file (if DWI in NRRD).")
    parser.add_argument('--output_bvec', help="Path to save the output b-vectors file (if DWI in NRRD).")
    parser.set_defaults(func=run_nrrd_to_nifti)

def run_nrrd_to_nifti(args):
    print(f"Converting NRRD file: {args.input_nrrd} to NIfTI: {args.output_nifti}")
    try:
        image_data, affine, bvals, bvecs, _ = read_nrrd_data(args.input_nrrd)

        if image_data is None:
            print(f"Failed to read data from NRRD file: {args.input_nrrd}", file=sys.stderr)
            sys.exit(1)

        # The data from nrrd.read is typically (ax0, ax1, ax2, ...).
        # NIfTI typically expects (i, j, k, ...) which is often (x, y, z, ...).
        # A common convention for NRRD data read by pynrrd is (depth, height, width),
        # which corresponds to (z, y, x) if thinking in terms of typical NIfTI display.
        # If the affine correctly maps voxel indices to world space, the data orientation
        # passed to Nifti1Image should align with what the affine expects.
        # However, NIfTI viewers often assume specific orientations (e.g. RAS for data array).
        # For simplicity, we save as is and rely on affine. If data needs reordering to a
        # canonical NIfTI orientation (e.g. RAS), that's a more complex step.
        # The affine from read_nrrd_data is constructed to map NRRD voxel coords to world.

        # Ensure data is float32 for NIfTI saving if it's not already, common practice.
        if not np.issubdtype(image_data.dtype, np.floating):
            image_data = image_data.astype(np.float32)

        save_nifti_data(image_data, affine, args.output_nifti) # Uses nib.save internally

        if bvals is not None and args.output_bval:
            np.savetxt(args.output_bval, bvals.reshape(1, -1), fmt='%g')
            print(f"b-values saved to: {args.output_bval}")
        if bvecs is not None and args.output_bvec:
            np.savetxt(args.output_bvec, bvecs, fmt='%.8f') # Assuming Nx3 format
            print(f"b-vectors saved to: {args.output_bvec}")

        print("NRRD to NIfTI conversion successful.")
    except Exception as e:
        print(f"Error during NRRD to NIfTI conversion: {e}", file=sys.stderr)
        sys.exit(1)


def setup_nifti_to_nrrd_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--input_nifti', required=True, help="Path to the input NIfTI file.")
    parser.add_argument('--output_nrrd', required=True, help="Path to save the output NRRD file.")
    parser.add_argument('--input_bval', help="Path to the input b-values file (for DWI).")
    parser.add_argument('--input_bvec', help="Path to the input b-vectors file (for DWI).")
    # Add options for nrrd_header_options like encoding, endian, etc.
    parser.add_argument('--nrrd_encoding', default='gzip', help="Encoding for NRRD file (e.g., 'raw', 'gzip', 'bz2'). Default: gzip.")
    parser.add_argument('--nrrd_endian', default='little', choices=['little', 'big'], help="Endianness for NRRD file. Default: little.")
    parser.set_defaults(func=run_nifti_to_nrrd)

def run_nifti_to_nrrd(args):
    print(f"Converting NIfTI file: {args.input_nifti} to NRRD: {args.output_nrrd}")
    try:
        img = nib.load(args.input_nifti)
        data = img.get_fdata()
        affine = img.affine

        bvals, bvecs = None, None
        if args.input_bval and args.input_bvec:
            if not os.path.exists(args.input_bval):
                raise FileNotFoundError(f"bval file not found: {args.input_bval}")
            if not os.path.exists(args.input_bvec):
                raise FileNotFoundError(f"bvec file not found: {args.input_bvec}")
            bvals = np.loadtxt(args.input_bval)
            bvecs = np.loadtxt(args.input_bvec)
            if bvecs.shape[0] == 3 and bvecs.shape[1] == len(bvals): # FSL format 3xN
                bvecs = bvecs.T
            print(f"Loaded b-values (count: {len(bvals)}) and b-vectors (shape: {bvecs.shape})")
        elif args.input_bval or args.input_bvec: # Only one provided
            print("Warning: Both bval and bvec files must be provided for DWI conversion. Proceeding without DWI info.", file=sys.stderr)


        nrrd_opts = {
            'encoding': args.nrrd_encoding,
            'endian': args.nrrd_endian
        }
        # Note: data from nib.load (get_fdata()) is usually in a specific orientation (RAS if header implies it).
        # The affine maps this to world. write_nrrd_data will use this affine.
        write_nrrd_data(
            output_filepath=args.output_nrrd,
            data=data,
            affine=affine,
            bvals=bvals,
            bvecs=bvecs,
            nrrd_header_options=nrrd_opts
        )
        print("NIfTI to NRRD conversion successful.")
    except Exception as e:
        print(f"Error during NIfTI to NRRD conversion: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Tools for converting between NIfTI and NRRD formats.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(title="Available Commands", dest="command_format_conversion")
    subparsers.required = True

    nrrd_to_nii_parser = subparsers.add_parser(
        "nrrd2nii",
        help="Convert NRRD file to NIfTI format.",
        description="Converts NRRD to NIfTI, optionally extracting DWI bvals/bvecs."
    )
    setup_nrrd_to_nifti_parser(nrrd_to_nii_parser)

    nii_to_nrrd_parser = subparsers.add_parser(
        "nii2nrrd",
        help="Convert NIfTI file to NRRD format.",
        description="Converts NIfTI to NRRD, optionally embedding DWI bvals/bvecs from text files."
    )
    setup_nifti_to_nrrd_parser(nii_to_nrrd_parser)

    # --- MHD to NIfTI Subcommand ---
    mhd_to_nii_parser = subparsers.add_parser(
        "mhd2nii",
        help="Convert MHD/MHA file to NIfTI format.",
        description="Converts MHD/MHA to NIfTI, optionally extracting DWI bvals/bvecs."
    )
    setup_mhd_to_nifti_parser(mhd_to_nii_parser)

    # --- NIfTI to MHD Subcommand ---
    nii_to_mhd_parser = subparsers.add_parser(
        "nii2mhd",
        help="Convert NIfTI file to MHD/MHA format.",
        description="Converts NIfTI to MHD/MHA, optionally embedding DWI bvals/bvecs."
    )
    setup_nifti_to_mhd_parser(nii_to_mhd_parser)

    # --- Analyze to NIfTI Subcommand ---
    analyze_to_nii_parser = subparsers.add_parser(
        "analyze2nii",
        help="Convert Analyze 7.5 file (.hdr/.img) to NIfTI format.",
        description="Converts Analyze 7.5 to NIfTI. DWI information is not handled by Analyze format."
    )
    setup_analyze_to_nifti_parser(analyze_to_nii_parser)

    # --- NIfTI to Analyze Subcommand ---
    nii_to_analyze_parser = subparsers.add_parser(
        "nii2analyze",
        help="Convert NIfTI file to Analyze 7.5 format (.hdr/.img).",
        description="Converts NIfTI to Analyze 7.5. DWI information is not stored in Analyze."
    )
    setup_nifti_to_analyze_parser(nii_to_analyze_parser)

    if len(sys.argv) <= 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        print(f"No function associated with command: {args.command_format_conversion}", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()


# --- MHD to NIfTI ---
def setup_mhd_to_nifti_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--input_mhd', required=True, help="Path to the input MHD/MHA file.")
    parser.add_argument('--output_nifti', required=True, help="Path to save the output NIfTI file.")
    parser.add_argument('--output_bval', help="Path to save the output b-values file (if DWI in MHD).")
    parser.add_argument('--output_bvec', help="Path to save the output b-vectors file (if DWI in MHD).")
    parser.set_defaults(func=run_mhd_to_nifti)

def run_mhd_to_nifti(args):
    print(f"Converting MHD/MHA file: {args.input_mhd} to NIfTI: {args.output_nifti}")
    try:
        image_data, affine, bvals, bvecs, _ = read_mhd_data(args.input_mhd)

        if image_data is None:
            print(f"Failed to read data from MHD/MHA file: {args.input_mhd}", file=sys.stderr)
            sys.exit(1)

        if not np.issubdtype(image_data.dtype, np.floating):
            image_data = image_data.astype(np.float32)

        save_nifti_data(image_data, affine, args.output_nifti)

        if bvals is not None and args.output_bval:
            np.savetxt(args.output_bval, bvals.reshape(1, -1), fmt='%g')
            print(f"b-values saved to: {args.output_bval}")
        if bvecs is not None and args.output_bvec:
            np.savetxt(args.output_bvec, bvecs, fmt='%.8f')
            print(f"b-vectors saved to: {args.output_bvec}")

        print("MHD/MHA to NIfTI conversion successful.")
    except Exception as e:
        print(f"Error during MHD/MHA to NIfTI conversion: {e}", file=sys.stderr)
        sys.exit(1)

# --- NIfTI to MHD ---
def setup_nifti_to_mhd_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--input_nifti', required=True, help="Path to the input NIfTI file.")
    parser.add_argument('--output_mhd', required=True, help="Path to save the output MHD/MHA file.")
    parser.add_argument('--input_bval', help="Path to the input b-values file (for DWI).")
    parser.add_argument('--input_bvec', help="Path to the input b-vectors file (for DWI).")
    # MHD does not have explicit encoding/endian like NRRD in its header structure for sitk.WriteImage options
    parser.set_defaults(func=run_nifti_to_mhd)

def run_nifti_to_mhd(args):
    print(f"Converting NIfTI file: {args.input_nifti} to MHD/MHA: {args.output_mhd}")
    try:
        img = nib.load(args.input_nifti)
        data = img.get_fdata()
        affine = img.affine

        bvals, bvecs = None, None
        if args.input_bval and args.input_bvec:
            if not os.path.exists(args.input_bval):
                raise FileNotFoundError(f"bval file not found: {args.input_bval}")
            if not os.path.exists(args.input_bvec):
                raise FileNotFoundError(f"bvec file not found: {args.input_bvec}")
            bvals = np.loadtxt(args.input_bval)
            bvecs = np.loadtxt(args.input_bvec)
            if bvecs.shape[0] == 3 and bvecs.shape[1] == len(bvals): # FSL format 3xN
                bvecs = bvecs.T
            print(f"Loaded b-values (count: {len(bvals)}) and b-vectors (shape: {bvecs.shape})")
        elif args.input_bval or args.input_bvec:
            print("Warning: Both bval and bvec files must be provided for DWI conversion. Proceeding without DWI info.", file=sys.stderr)

        write_mhd_data(
            output_filepath=args.output_mhd,
            data=data,
            affine=affine,
            bvals=bvals,
            bvecs=bvecs
        )
        print("NIfTI to MHD/MHA conversion successful.")
    except Exception as e:
        print(f"Error during NIfTI to MHD/MHA conversion: {e}", file=sys.stderr)
        sys.exit(1)


# --- Analyze to NIfTI ---
def setup_analyze_to_nifti_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--input_analyze', required=True, help="Path to the input Analyze file (.hdr or .img).")
    parser.add_argument('--output_nifti', required=True, help="Path to save the output NIfTI file.")
    parser.set_defaults(func=run_analyze_to_nifti)

def run_analyze_to_nifti(args):
    print(f"Converting Analyze file: {args.input_analyze} to NIfTI: {args.output_nifti}")
    try:
        image_data, affine, _ = read_analyze_data(args.input_analyze) # header also returned but not used here

        if image_data is None:
            print(f"Failed to read data from Analyze file: {args.input_analyze}", file=sys.stderr)
            sys.exit(1)

        # Data should already be float32 from read_analyze_data
        save_nifti_data(image_data, affine, args.output_nifti) # Uses nib.Nifti1Image and nib.save

        print("Analyze to NIfTI conversion successful.")
    except Exception as e:
        print(f"Error during Analyze to NIfTI conversion: {e}", file=sys.stderr)
        sys.exit(1)

# --- NIfTI to Analyze ---
def setup_nifti_to_analyze_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--input_nifti', required=True, help="Path to the input NIfTI file.")
    parser.add_argument('--output_analyze', required=True, help="Path to save the output Analyze file (.hdr/.img).")
    parser.set_defaults(func=run_nifti_to_analyze)

def run_nifti_to_analyze(args):
    print(f"Converting NIfTI file: {args.input_nifti} to Analyze: {args.output_analyze}")
    try:
        img = nib.load(args.input_nifti)
        data = img.get_fdata(dtype=np.float32) # Ensure float32 for Analyze writing
        affine = img.affine
        # Analyze header is minimal; specific information might be lost.
        # Pass original NIfTI header to AnalyzeImage to preserve what's possible (e.g. pixdim).
        # However, AnalyzeImage will construct its own AnalyzeHeader.
        # For more control, one might create an AnalyzeHeader manually.
        write_analyze_data(
            output_filepath=args.output_analyze,
            data=data,
            affine=affine,
            header=img.header # Pass NIfTI header, AnalyzeImage will adapt
        )
        print("NIfTI to Analyze conversion successful.")
    except Exception as e:
        print(f"Error during NIfTI to Analyze conversion: {e}", file=sys.stderr)
        sys.exit(1)
