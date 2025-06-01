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
    'read_analyze_data',           # Added from analyze_utils
    'write_analyze_data'           # Added from analyze_utils
]

# Import from dicom_utils
try:
    from .dicom_utils import (
        convert_dicom_to_nifti_main,
        convert_dwi_dicom_to_nifti,
        anonymize_dicom_file,
        anonymize_dicom_directory
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
