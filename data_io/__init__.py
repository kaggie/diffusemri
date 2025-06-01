# __init__.py for diffusemri.data_io

from .nifti import (
    load_nifti_dwi,
    load_nifti_mask,
    load_fsl_bvecs,
    load_fsl_bvals,
    load_nifti_study
)

from .gradients import (
    create_gradient_table
)

# Optionally, define __all__ to specify public API
__all__ = [
    'load_nifti_dwi',
    'load_nifti_mask',
    'load_fsl_bvecs',
    'load_fsl_bvals',
    'load_nifti_study',
    'create_gradient_table',
    'convert_dicom_to_nifti_main',
    'convert_dwi_dicom_to_nifti',
    'anonymize_dicom_file',
    'anonymize_dicom_directory',
    'read_nrrd_data',
    'write_nrrd_data',
    'read_mhd_data',
    'write_mhd_data',
    'read_analyze_data',
    'write_analyze_data',
    'read_ismrmrd_file',
    'convert_ismrmrd_to_nifti_and_metadata',
    'read_parrec_data',
    'convert_parrec_to_nifti',
    'save_dict_to_hdf5',
    'load_dict_from_hdf5',
    'save_dict_to_mat',
    'load_dict_from_mat',
    'write_nifti_to_dicom_secondary' # Added from dicom_utils
]

# Import from dicom_utils
try:
    from .dicom_utils import (
        convert_dicom_to_nifti_main,
        convert_dwi_dicom_to_nifti,
        anonymize_dicom_file,
        anonymize_dicom_directory,
        write_nifti_to_dicom_secondary # Ensure it's imported
    )
except ImportError:
    pass # Handled by top-level __init__ or direct use

# Import from nrrd_utils
try:
    from .nrrd_utils import read_nrrd_data, write_nrrd_data
except ImportError:
    pass

# Import from mhd_utils
try:
    from .mhd_utils import read_mhd_data, write_mhd_data
except ImportError:
    pass

# Import from analyze_utils
try:
    from .analyze_utils import read_analyze_data, write_analyze_data
except ImportError:
    pass

# Import from ismrmrd_utils
try:
    from .ismrmrd_utils import read_ismrmrd_file, convert_ismrmrd_to_nifti_and_metadata
except ImportError:
    pass

# Import from parrec_utils
try:
    from .parrec_utils import read_parrec_data, convert_parrec_to_nifti
except ImportError:
    pass

# Import from generic_utils
try:
    from .generic_utils import (
        save_dict_to_hdf5, load_dict_from_hdf5,
        save_dict_to_mat, load_dict_from_mat
    )
except ImportError:
    pass
