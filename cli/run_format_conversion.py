import argparse
import sys
import os
import numpy as np
import nibabel as nib
# Assuming nrrd_utils is in data_io, and cli_utils for general helpers
try:
    from diffusemri.data_io.nrrd_utils import read_nrrd_data, write_nrrd_data
    from diffusemri.data_io.mhd_utils import read_mhd_data, write_mhd_data
    from diffusemri.data_io.analyze_utils import read_analyze_data, write_analyze_data
    from diffusemri.data_io.ismrmrd_utils import convert_ismrmrd_to_nifti_and_metadata
    from diffusemri.data_io.parrec_utils import convert_parrec_to_nifti
    from diffusemri.data_io.dicom_utils import write_nifti_to_dicom_secondary # Added for nii2dicom_sec
    from diffusemri.data_io.bruker_utils import read_bruker_dwi_data # Added for Bruker
    from diffusemri.data_io.cli_utils import save_nifti_data
except ImportError:
    # Fallback for direct script execution if diffusemri is not in PYTHONPATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from data_io.nrrd_utils import read_nrrd_data, write_nrrd_data
    from data_io.mhd_utils import read_mhd_data, write_mhd_data
    from data_io.analyze_utils import read_analyze_data, write_analyze_data
    from data_io.ismrmrd_utils import convert_ismrmrd_to_nifti_and_metadata
    from data_io.parrec_utils import convert_parrec_to_nifti
    from data_io.dicom_utils import write_nifti_to_dicom_secondary # Added for nii2dicom_sec
    from data_io.bruker_utils import read_bruker_dwi_data # Added for Bruker
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

    # --- ISMRMRD to NIfTI Subcommand (Placeholder) ---
    ismrmrd_to_nii_parser = subparsers.add_parser(
        "ismrmrd_convert",
        help="Convert ISMRMRD file to NIfTI format (placeholder).",
        description="Converts ISMRMRD to NIfTI and extracts metadata. This is currently a placeholder."
    )
    setup_ismrmrd_convert_parser(ismrmrd_to_nii_parser)

    # --- PAR/REC to NIfTI Subcommand ---
    parrec_to_nii_parser = subparsers.add_parser(
        "parrec2nii", # Alias for parrec_to_nifti
        help="Convert Philips PAR/REC file to NIfTI format.",
        description="Converts PAR/REC to NIfTI, extracting DWI bvals/bvecs if available."
    )
    setup_parrec_to_nifti_parser(parrec_to_nii_parser)

    # --- NIfTI to DICOM Secondary Capture Subcommand ---
    nii_to_dicom_sec_parser = subparsers.add_parser(
        "nii2dicom_sec",
        help="Convert NIfTI file to DICOM Secondary Capture series.",
        description="Converts a NIfTI volume into a series of DICOM Secondary Capture files."
    )
    setup_nifti_to_dicom_secondary_parser(nii_to_dicom_sec_parser)

    # --- Bruker to NIfTI Subcommand ---
    bruker_to_nii_parser = subparsers.add_parser(
        "bruker2nii", # Alias
        aliases=['bruker_to_nifti'],
        help="Convert Bruker ParaVision DWI data to NIfTI format.",
        description="Reads a Bruker ParaVision experiment directory and converts DWI data to NIfTI, bval, and bvec files."
    )
    setup_bruker_to_nifti_parser(bruker_to_nii_parser)

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


# --- ISMRMRD to NIfTI (Placeholder) ---
def setup_ismrmrd_convert_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--input_ismrmrd', required=True, help="Path to the input ISMRMRD file (.h5).")
    parser.add_argument('--output_base', required=True, help="Base path/filename for output NIfTI and metadata files (e.g., 'output/my_scan').")
    parser.set_defaults(func=run_ismrmrd_convert)

def run_ismrmrd_convert(args):
    print(f"Attempting ISMRMRD conversion for: {args.input_ismrmrd}")
    print("Note: This functionality is currently a placeholder and not fully implemented.")
    success = convert_ismrmrd_to_nifti_and_metadata(args.input_ismrmrd, args.output_base)
    if success:
        print(f"Placeholder ISMRMRD conversion 'completed' for base: {args.output_base}")
    else:
        print(f"Placeholder ISMRMRD conversion 'failed' for base: {args.output_base} (as expected for placeholder).")
        # sys.exit(1) # Do not exit with error for a placeholder


# --- PAR/REC to NIfTI ---
def setup_parrec_to_nifti_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--input_parrec', required=True, help="Path to the input PAR or REC file.")
    parser.add_argument('--output_nifti', required=True, help="Path to save the output NIfTI file.")
    parser.add_argument('--output_bval', help="Path to save the output b-values file (if DWI).")
    parser.add_argument('--output_bvec', help="Path to save the output b-vectors file (if DWI).")
    parser.add_argument('--strict_sort', action='store_true', default=True,
                        help="Use strict volume sorting for PAR/REC (default: True).")
    parser.add_argument('--no_strict_sort', action='store_false', dest='strict_sort',
                        help="Disable strict volume sorting for PAR/REC.")
    parser.add_argument('--scaling_method', default='dv', choices=['dv', 'fp'],
                        help="Scaling method for PAR/REC data ('dv' or 'fp', default: 'dv').")
    parser.set_defaults(func=run_parrec_to_nifti)

def run_parrec_to_nifti(args):
    print(f"Converting PAR/REC file: {args.input_parrec} to NIfTI: {args.output_nifti}")
    try:
        success = convert_parrec_to_nifti(
            parrec_filepath=args.input_parrec,
            output_nifti_file=args.output_nifti,
            output_bval_file=args.output_bval,
            output_bvec_file=args.output_bvec,
            strict_sort=args.strict_sort,
            scaling=args.scaling_method
        )
        if success:
            print("PAR/REC to NIfTI conversion successful.")
        else:
            print("PAR/REC to NIfTI conversion failed.", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error during PAR/REC to NIfTI conversion: {e}", file=sys.stderr)
        sys.exit(1)


# --- NIfTI to DICOM Secondary Capture ---
def setup_nifti_to_dicom_secondary_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--input_nifti', required=True, help="Path to the input NIfTI file.")
    parser.add_argument('--output_dicom_dir', required=True, help="Path to the output directory for DICOM slices.")
    parser.add_argument('--series_description', default="Processed NIfTI Image", help="Series Description for DICOM tags.")
    parser.add_argument('--rescale_slope', type=float, default=1.0, help="Rescale Slope for pixel data scaling.")
    parser.add_argument('--rescale_intercept', type=float, default=0.0, help="Rescale Intercept for pixel data scaling.")
    parser.add_argument('--window_center', type=float, help="Window Center for DICOM display.")
    parser.add_argument('--window_width', type=float, help="Window Width for DICOM display.")
    parser.set_defaults(func=run_nifti_to_dicom_secondary)

def run_nifti_to_dicom_secondary(args):
    print(f"Converting NIfTI file: {args.input_nifti} to DICOM Secondary Capture series in: {args.output_dicom_dir}")
    try:
        success = write_nifti_to_dicom_secondary(
            nifti_filepath=args.input_nifti,
            output_dicom_dir=args.output_dicom_dir,
            series_description=args.series_description,
            rescale_slope=args.rescale_slope,
            rescale_intercept=args.rescale_intercept,
            window_center=args.window_center,
            window_width=args.window_width
        )
        if success:
            print("NIfTI to DICOM Secondary Capture conversion successful.")
        else:
            print("NIfTI to DICOM Secondary Capture conversion failed.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error during NIfTI to DICOM Secondary Capture conversion: {e}", file=sys.stderr)
        sys.exit(1)


# --- Bruker to NIfTI ---
def setup_bruker_to_nifti_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--input_bruker_dir', required=True,
                        help="Path to the Bruker ParaVision experiment directory (e.g., subject/study/exp_num).")
    parser.add_argument('--output_nifti', required=True, help="Path to save the output NIfTI file.")
    parser.add_argument('--output_bval', required=True, help="Path to save the output b-values file.")
    parser.add_argument('--output_bvec', required=True, help="Path to save the output b-vectors file.")
    parser.set_defaults(func=run_bruker_to_nifti)

def run_bruker_to_nifti(args):
    print(f"Converting Bruker ParaVision data from: {args.input_bruker_dir}")
    print(f"Output NIfTI: {args.output_nifti}")
    print(f"Output bval: {args.output_bval}")
    print(f"Output bvec: {args.output_bvec}")

    try:
        result = read_bruker_dwi_data(args.input_bruker_dir)
        if result is None:
            print(f"Failed to read Bruker data from {args.input_bruker_dir}", file=sys.stderr)
            sys.exit(1)

        image_data, affine, bvals, bvecs, metadata_dict = result

        if image_data is None or affine is None:
            print("No image data or affine extracted from Bruker dataset.", file=sys.stderr)
            sys.exit(1)

        if not np.issubdtype(image_data.dtype, np.floating):
            image_data = image_data.astype(np.float32) # Ensure float for NIfTI saving

        save_nifti_data(image_data, affine, args.output_nifti)
        print(f"NIfTI file saved to: {args.output_nifti}")

        if bvals is not None:
            np.savetxt(args.output_bval, bvals.reshape(1, -1), fmt='%g')
            print(f"b-values saved to: {args.output_bval}")
        else:
            print("Warning: b-values were not extracted. .bval file will not be created or will be empty.", file=sys.stderr)

        if bvecs is not None:
            np.savetxt(args.output_bvec, bvecs, fmt='%.8f') # Assuming Nx3 format
            print(f"b-vectors saved to: {args.output_bvec}")
        else:
            print("Warning: b-vectors were not extracted. .bvec file will not be created or will be empty.", file=sys.stderr)

        print("Bruker ParaVision to NIfTI conversion successful.")

    except Exception as e:
        print(f"Error during Bruker to NIfTI conversion: {e}", file=sys.stderr)
        sys.exit(1)
