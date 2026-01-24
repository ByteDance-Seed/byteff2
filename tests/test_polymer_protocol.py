"""
Unit tests for polymer electrolyte protocol.
"""

import pytest
import tempfile
import os
import json

from byteff2.toolkit.polymer_builder import (
    PolymerChainBuilder,
    PEOBuilder,
    PolymerLibrary,
)
from byteff2.toolkit.box_builders import (
    BoxBuilder,
    PackmolBoxBuilder,
    GMXBoxBuilder,
)
from byteff2.toolkit.polymer_protocol import (
    PolymerComponent,
    PolymerType,
    PolymerElectrolyteProtocol,
)
from byteff2.toolkit.polymer_simulation import (
    PolymerEquilibrationProtocol,
    EquilibrationStage,
)
from byteff2.toolkit.config_schemas import (
    validate_config,
    get_example_config,
    EXAMPLE_PEO_LITFSI,
)


class TestPolymerChainBuilder:
    """Tests for polymer chain building."""
    
    def test_peo_builder_creation(self):
        """Test PEO builder initialization."""
        builder = PEOBuilder(dp=10)
        assert builder.dp == 10
        assert builder.monomer_smiles == "[*]OCC[*]"
    
    def test_generic_polymer_builder(self):
        """Test generic polymer builder."""
        builder = PolymerChainBuilder(
            monomer_smiles="[*]CC[*]",  # Polyethylene
            dp=20,
            end_group_left="C",
            end_group_right="C"
        )
        assert builder.dp == 20
    
    def test_polymer_library(self):
        """Test polymer library has common polymers."""
        assert 'PEO' in PolymerLibrary.POLYMERS
        assert 'PPO' in PolymerLibrary.POLYMERS
        assert 'PVDF' in PolymerLibrary.POLYMERS
    
    @pytest.mark.skipif(
        not pytest.importorskip("rdkit", reason="RDKit required"),
        reason="RDKit not available"
    )
    def test_peo_chain_generation(self):
        """Test actual PEO chain generation."""
        builder = PEOBuilder(dp=5)
        chain = builder.build_chain()
        
        # PEO: (CH2-CH2-O)n
        # Each unit has 7 atoms (2C + 4H + 1O), plus end groups
        assert chain is not None


class TestBoxBuilders:
    """Tests for box building backends."""
    
    def test_gmx_builder_no_polymer_support(self):
        """Test GMX builder reports no polymer support."""
        builder = GMXBoxBuilder()
        assert builder.supports_polymers() == False
    
    def test_packmol_builder_polymer_support(self):
        """Test Packmol builder supports polymers."""
        builder = PackmolBoxBuilder()
        assert builder.supports_polymers() == True
    
    def test_packmol_input_generation(self):
        """Test Packmol input file generation."""
        builder = PackmolBoxBuilder()
        
        # Mock components
        class MockComponent:
            def __init__(self, name, count, pdb_path):
                self.name = name
                self.molar_num = count
                self.pdb_path = pdb_path
        
        components = [
            MockComponent("PEO", 10, "/tmp/peo.pdb"),
            MockComponent("LI", 50, "/tmp/li.pdb"),
        ]
        
        packmol_input = builder._generate_packmol_input(
            components, 
            box_size=5.0,
            output_dir="/tmp"
        )
        
        assert "tolerance" in packmol_input
        assert "PEO" in packmol_input or "peo.pdb" in packmol_input


class TestPolymerComponent:
    """Tests for PolymerComponent class."""
    
    def test_polymer_component_creation(self):
        """Test creating a polymer component."""
        comp = PolymerComponent(
            name="PEO",
            smiles="COCCOCCOCCO",
            molar_num=10,
            polymer_type=PolymerType.LINEAR,
            dp=50,
            monomer_smiles="CCO"
        )
        
        assert comp.name == "PEO"
        assert comp.polymer_type == PolymerType.LINEAR
        assert comp.degree_of_polymerization == 50


class TestPolymerEquilibrationProtocol:
    """Tests for equilibration protocol."""
    
    def test_default_stages(self):
        """Test default equilibration stages."""
        protocol = PolymerEquilibrationProtocol(
            temperature=363,
            pressure=1.0
        )
        
        assert len(protocol.stages) > 0
        stage_names = [s.name for s in protocol.stages]
        assert 'minimize' in stage_names
        assert any('nvt' in name for name in stage_names)
        assert any('npt' in name for name in stage_names)
    
    def test_custom_stages(self):
        """Test custom equilibration stages."""
        custom_stages = [
            {'name': 'minimize', 'ensemble': 'minimize', 'steps': 1000},
            {'name': 'nvt_short', 'ensemble': 'nvt', 'steps': 5000},
        ]
        
        protocol = PolymerEquilibrationProtocol(
            temperature=300,
            stages=custom_stages
        )
        
        assert len(protocol.stages) == 2
    
    def test_mdp_generation(self):
        """Test MDP file content generation."""
        protocol = PolymerEquilibrationProtocol(temperature=363)
        
        stage = EquilibrationStage(
            name='test_nvt',
            ensemble='nvt',
            steps=10000,
            temperature=363
        )
        
        mdp = protocol._generate_nvt_mdp(stage)
        
        assert 'integrator' in mdp
        assert '363' in mdp
        assert 'nsteps' in mdp


class TestConfigSchemas:
    """Tests for configuration validation."""
    
    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = EXAMPLE_PEO_LITFSI.copy()
        assert validate_config(config) == True
    
    def test_missing_required_field(self):
        """Test validation fails with missing field."""
        config = EXAMPLE_PEO_LITFSI.copy()
        del config['temperature']
        
        with pytest.raises(ValueError):
            validate_config(config)
    
    def test_get_example_config(self):
        """Test getting example configurations."""
        config = get_example_config('PEO_LiTFSI')
        
        assert 'protocol' in config
        assert 'temperature' in config
        assert 'components' in config
        assert 'PEO' in config['components']
    
    def test_unknown_example(self):
        """Test error on unknown example."""
        with pytest.raises(ValueError, match="Unknown system"):
            get_example_config('UNKNOWN_SYSTEM')


class TestPolymerElectrolyteProtocol:
    """Integration tests for full polymer protocol."""
    
    def test_protocol_initialization(self):
        """Test protocol can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol = PolymerElectrolyteProtocol(
                params_dir=os.path.join(tmpdir, 'params'),
                output_dir=os.path.join(tmpdir, 'output')
            )
            
            assert protocol is not None
    
    def test_protocol_with_config(self):
        """Test protocol with configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol = PolymerElectrolyteProtocol(
                params_dir=os.path.join(tmpdir, 'params'),
                output_dir=os.path.join(tmpdir, 'output')
            )
            
            config = get_example_config('PEO_LiTFSI')
            config['output_dir'] = os.path.join(tmpdir, 'output')
            config['params_dir'] = os.path.join(tmpdir, 'params')
            
            protocol.config = config
            
            # Check components are parsed correctly
            assert protocol.config['temperature'] == 363
    
    def test_is_polymer_system_detection(self):
        """Test detection of polymer systems."""
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol = PolymerElectrolyteProtocol(
                params_dir=os.path.join(tmpdir, 'params'),
                output_dir=os.path.join(tmpdir, 'output')
            )
            
            # Polymer system
            polymer_config = {'components': {'PEO': {'type': 'polymer'}}}
            assert protocol._is_polymer_system(polymer_config) == True
            
            # Non-polymer system
            liquid_config = {'components': {'EC': {'type': 'solvent'}}}
            assert protocol._is_polymer_system(liquid_config) == False


@pytest.fixture
def sample_config():
    """Provide sample configuration for tests."""
    return get_example_config('PEO_LiTFSI')


def test_end_to_end_config_validation(sample_config):
    """Test end-to-end configuration workflow."""
    # Modify for testing
    sample_config['npt_steps'] = 100  # Very short for testing
    sample_config['nvt_steps'] = 100
    
    assert validate_config(sample_config)
    assert sample_config['protocol'] == 'PolymerTransport'
    assert 'PEO' in sample_config['components']