import argparse
import sys
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(
        description="Main command-line interface for the diffusemri library.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--version', 
        action='version', 
        version='%(prog)s 0.1.0', # Placeholder version
        help="Show program's version number and exit."
    )
    
    subparsers = parser.add_subparsers(
        title="Available Commands", 
        dest="command",
        # In Python 3.7+, required=True can be used directly in add_subparsers.
        # For broader compatibility or if issues arise, check after parsing.
        help="Use '<command> -h' for more information on a specific command."
    )
    # Make subparsers required by checking if args.command is None after parsing if not using required=True
    # However, the logic below with parse_known_args and explicit checks handles no command.

    # --- Preprocessing Command ---
    preprocess_parser = subparsers.add_parser(
        "preprocess", 
        help="Access preprocessing tools (masking, denoising, FSL eddy correction). "
             "Type 'diffusemri_cli.py preprocess -h' for its subcommands.",
        description="Provides access to various dMRI preprocessing tools. "
                    "This command acts as an entry point to 'cli/run_preprocessing.py'.\n"
                    "Example: diffusemri_cli.py preprocess masking --dwi ...",
        add_help=False # Let the subprocess handle its own detailed help
    )
    
    # --- DTI Fitting Command ---
    fit_dti_parser = subparsers.add_parser(
        "fit_dti", 
        help="Fit DTI model to DWI data. Type 'diffusemri_cli.py fit_dti -h' for options.",
        description="Fits the DTI model. This command passes arguments to 'cli/run_dti_fit.py'.",
        add_help=False 
    )

    # --- NODDI Fitting Command ---
    fit_noddi_parser = subparsers.add_parser(
        "fit_noddi", 
        help="Fit NODDI model to DWI data. Type 'diffusemri_cli.py fit_noddi -h' for options.",
        description="Fits the NODDI model. This command passes arguments to 'cli/run_noddi_fit.py'.",
        add_help=False
    )

    # --- Tracking Command ---
    track_parser = subparsers.add_parser(
        "track",
        help="Access tractography tools. Type 'diffusemri_cli.py track -h' for options.",
        description="Provides access to tractography algorithms. This command passes arguments to 'cli/run_tracking.py'.",
        add_help=False
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    # Use parse_known_args to separate the main command and its direct args (none here)
    # from the rest of the arguments that will be passed to the sub-script.
    args, remaining_argv = parser.parse_known_args()

    if args.command is None: # If no command was recognized by main parser (e.g. only --version, or invalid command)
        # If --version was processed, argparse exits. If not, and no command, it might error or print help.
        # This check is for safety if required=True on subparsers isn't effective or used.
        # Argparse with required subparsers usually handles this.
        # If parse_known_args returns a command, it was valid at this level.
        # If user typed "diffusemri_cli.py -h" for preprocess, args.command is "preprocess", remaining_argv is ["-h"]
        pass # Let it proceed to script execution, sub-script will handle its help.


    script_name = None
    # Determine the path to the 'cli' directory relative to this script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path_dir = os.path.join(base_dir, 'cli')

    if args.command == "preprocess":
        script_name = "run_preprocessing.py"
    elif args.command == "fit_dti":
        script_name = "run_dti_fit.py"
    elif args.command == "fit_noddi":
        script_name = "run_noddi_fit.py"
    elif args.command == "track":
        script_name = "run_tracking.py"
    else:
        # This case should be hit if a valid command was parsed by main parser,
        # but not matched in the if/elif chain. This indicates a logic error here.
        # However, if args.command was None and not caught by `required=True` on subparsers.
        # Argparse usually exits before this if command is truly unknown.
        parser.print_help(sys.stderr)
        print(f"Error: Unknown or no command specified '{args.command}'.", file=sys.stderr)
        sys.exit(1)

    full_script_path = os.path.join(script_path_dir, script_name)

    if not os.path.exists(full_script_path):
        print(f"Error: Sub-script not found: {full_script_path}", file=sys.stderr)
        print("Please ensure the 'cli' directory and its scripts are correctly placed.")
        sys.exit(1)

    command_to_run = [sys.executable, full_script_path] + remaining_argv
    
    if '-h' in remaining_argv or '--help' in remaining_argv:
        print(f"Displaying help for command '{args.command}' (from script '{script_name}'):")
    else:
        print(f"Executing command '{args.command}' via script '{script_name}' with arguments: {' '.join(remaining_argv)}")
    
    try:
        # text=True for Python 3.7+ to handle stdout/stderr as strings
        process = subprocess.run(command_to_run, check=True, text=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{args.command}' (script '{script_name}') failed with exit code {e.returncode}.", file=sys.stderr)
        # Sub-script's stderr should have already been printed.
        sys.exit(e.returncode)
    except FileNotFoundError: 
        print(f"Error: Could not execute sub-script. Ensure Python is in PATH and script exists: {full_script_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e: 
        print(f"An unexpected error occurred while trying to run '{script_name}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
```
