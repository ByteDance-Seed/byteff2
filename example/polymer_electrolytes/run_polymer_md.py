"""
Example script for running polymer electrolyte MD simulations.

Usage:
    python run_polymer_md.py config.json
    
Or use the provided example configurations:
    python run_polymer_md.py --example PEO_LiTFSI
"""

import argparse
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from byteff2.toolkit.polymer_protocol import PolymerElectrolyteProtocol, PolymerTransportProtocol, PolymerDensityProtocol
# from byteff2.toolkit.polymer_simulation import PolymerTransportProtocol
from byteff2.toolkit.config_schemas import get_example_config, validate_config


def main():
    parser = argparse.ArgumentParser(
        description='Run polymer electrolyte MD simulations with ByteFF2'
    )
    parser.add_argument(
        '--config',
        nargs='?',
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--example',
        choices=['PEO_LiTFSI', 'PEO_LiFSI', 'PPO_LiTFSI', 'block_copolymer'],
        help='Use an example configuration'
    )
    parser.add_argument(
        '--output-dir',
        default='./output',
        help='Output directory (default: ./output)'
    )
    parser.add_argument(
        '--params-dir',
        default='./params',
        help='Parameters directory (default: ./params)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration without running simulation'
    )
    parser.add_argument(
        '--skip-post-process',
        action='store_true',
        help='Skip post-processing step'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.example:
        config = get_example_config(args.example)
        print(f"Using example configuration: {args.example}")
    elif args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        parser.print_help()
        sys.exit(1)
    
    # Override directories if specified
    config['output_dir'] = args.output_dir
    config['params_dir'] = args.params_dir
    
    # Validate
    try:
        validate_config(config)
        print("Configuration validated successfully.")
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    if args.dry_run:
        print("\nConfiguration:")
        print(json.dumps(config, indent=2))
        return
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['params_dir'], exist_ok=True)
    
    # Save configuration
    config_output = os.path.join(config['output_dir'], 'config.json')
    with open(config_output, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_output}")
    
    # Run simulation
    protocol_name = config.get('protocol', 'PolymerTransport')
    
    if 'Transport' in protocol_name:
        protocol = PolymerTransportProtocol(
            output_dir=config['output_dir'],
            params_dir=config['params_dir']
        )
    elif 'Density' in protocol_name:
        protocol = PolymerDensityProtocol(
            output_dir=config['output_dir'],
            params_dir=config['params_dir']
        )
    else:
        protocol = PolymerElectrolyteProtocol(
            output_dir=config['output_dir'],
            params_dir=config['params_dir']
        )
    
    # protocol.config = config
    # IMPORTANT: Call setup_from_config to parse polymer components: 01-30-2026
    protocol.setup_from_config(config)
    
    print(f"\nStarting {protocol_name} simulation...")
    print(f"Temperature: {config['temperature']} K")
    print(f"Components: {list(config['components'].keys())}")
    print(f"Polymer components: {list(protocol.polymer_components.keys())}")
    
    try:
        # Run protocol (same method name as liquid electrolytes)
        protocol.run_protocol()
        
        # Post-process (unless skipped)
        if not args.skip_post_process:
            protocol.post_process()
        
        print("\nSimulation completed successfully!")
        
    except Exception as e:
        print(f"\nSimulation failed with error: {e}")
        raise


if __name__ == '__main__':
    main()