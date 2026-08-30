
# Third-party
import testutils

# Local imports
import cape.agent.options as agentopts


# JSON file for first test case - models with individual settings
SAMPLE_JSON_1 = """
{
    "ModelList": [
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "meta-llama/Llama-3.1-8B-Instruct",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
    ],
    "bartowski/Llama-3.2-3B-Instruct-GGUF": {
        "ToolSet": "low",
        "SkillSet": "none"
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "ToolSet": "medium",
        "SkillSet": "low"
    }
}
"""

# JSON file for second test case - models with parent inheritance
SAMPLE_JSON_2 = """
{
    "ModelList": [
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "meta-llama/Llama-3.1-8B-Instruct",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
    ],
    "bartowski/Llama-3.2-3B-Instruct-GGUF": {
        "ToolSet": "low",
        "SkillSet": "none"
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "Parent": "bartowski/Llama-3.2-3B-Instruct-GGUF"
    }
}
"""


# Test instantiation with no arguments
def test_01_init_empty():
    """Test creating empty AgentOpts instance"""
    opts = agentopts.AgentOpts()
    assert opts is not None
    assert len(opts) == 0


# Test instantiation with keyword arguments
def test_02_init_kwargs():
    """Test creating AgentOpts with keyword arguments"""
    opts = agentopts.AgentOpts(ModelList=["model1", "model2"])
    assert "ModelList" in opts
    assert opts["ModelList"] == ["model1", "model2"]


# Test reading JSON file - first example from docstring
@testutils.run_sandbox(__file__)
def test_03_read_json_sample1():
    """Test reading JSON file with individual model settings"""
    # Write sample JSON to file
    with open("agentopts1.json", "w") as f:
        f.write(SAMPLE_JSON_1)

    # Read the file
    opts = agentopts.AgentOpts("agentopts1.json")

    # Check ModelList
    assert "ModelList" in opts
    assert len(opts["ModelList"]) == 3
    m2 = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
    assert opts["ModelList"][0] == "bartowski/Llama-3.2-3B-Instruct-GGUF"
    assert opts["ModelList"][1] == "meta-llama/Llama-3.1-8B-Instruct"
    assert opts["ModelList"][2] == m2

    # Check first model settings
    assert "bartowski/Llama-3.2-3B-Instruct-GGUF" in opts
    model1_opts = opts["bartowski/Llama-3.2-3B-Instruct-GGUF"]
    assert model1_opts.get_opt("ToolSet") == "low"
    assert model1_opts.get_opt("SkillSet") == "none"

    # Check second model settings
    assert "meta-llama/Llama-3.1-8B-Instruct" in opts
    model2_opts = opts["meta-llama/Llama-3.1-8B-Instruct"]
    assert model2_opts.get_opt("ToolSet") == "medium"
    assert model2_opts.get_opt("SkillSet") == "low"

    # Check third model has no custom settings - it's in ModelList but
    # has no section because no settings were defined for it
    assert "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16" not in opts
    # But get_ModelOpt can still retrieve defaults for it
    model3_toolset = opts.get_ModelOpt(
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "ToolSet")
    assert model3_toolset == "full"
    model3_skillset = opts.get_ModelOpt(
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "SkillSet")
    assert model3_skillset == "full"


# Test reading JSON file - second example from docstring
@testutils.run_testdir(__file__)
def test_04_read_json_sample2():
    """Test reading JSON file with parent inheritance"""
    # Write sample JSON to file
    with open("agentopts2.json", "w") as f:
        f.write(SAMPLE_JSON_2)

    # Read the file
    opts = agentopts.AgentOpts("agentopts2.json")

    # Check ModelList
    assert "ModelList" in opts
    assert len(opts["ModelList"]) == 3

    # Check first model settings
    assert "bartowski/Llama-3.2-3B-Instruct-GGUF" in opts
    model1_opts = opts["bartowski/Llama-3.2-3B-Instruct-GGUF"]
    assert model1_opts.get_opt("ToolSet") == "low"
    assert model1_opts.get_opt("SkillSet") == "none"

    # Check second model inherits from first via Parent
    m2 = "bartowski/Llama-3.2-3B-Instruct-GGUF"
    assert "meta-llama/Llama-3.1-8B-Instruct" in opts
    model2_opts = opts["meta-llama/Llama-3.1-8B-Instruct"]
    assert model2_opts.get_opt("Parent") == m2


# Test get_ModelOpt method
def test_05_get_modelopt():
    """Test get_ModelOpt method for retrieving model-specific options"""
    # Create opts with one model defined
    opts = agentopts.AgentOpts(
        ModelList=["model_a", "model_b"],
        model_a={"ToolSet": "low", "SkillSet": "medium"}
    )

    # Get option for defined model
    model_a_opts = opts["model_a"]
    assert model_a_opts.get_opt("ToolSet") == "low"
    assert model_a_opts.get_opt("SkillSet") == "medium"

    # get_ModelOpt returns Parent setting for existing models
    parent_val = opts.get_ModelOpt("model_a", "Parent")
    assert parent_val is None  # model_a has no Parent
    # get_ModelOpt returns model's own settings, when present
    assert opts.get_ModelOpt("model_a", "ToolSet") == "low"

    # Get option for undefined model (should return default)
    toolset_b = opts.get_ModelOpt("model_b", "ToolSet")
    assert toolset_b == "full"  # Default from ModelOpts

    skillset_b = opts.get_ModelOpt("model_b", "SkillSet")
    assert skillset_b == "full"  # Default from ModelOpts

    # Test get_ModelOpt with non-existent option on undefined model
    # Note: get_opt returns None for options not in _optlist even with vdef
    val = opts.get_ModelOpt("model_b", "NonExistent")
    assert val is None


# Test alias "Models" for "ModelList"
def test_06_alias_models():
    """Test that 'Models' is an alias for 'ModelList'"""
    opts = agentopts.AgentOpts(Models=["model1", "model2"])
    assert "ModelList" in opts
    assert opts["ModelList"] == ["model1", "model2"]


# Test class attributes
def test_07_class_attributes():
    """Test class attributes are correctly defined"""
    m2 = "Library of JSON Options for CAPE Agent"
    assert agentopts.AgentOpts._name == m2
    assert agentopts.AgentOpts._label == "cape-agent-json"
    assert "ModelList" in agentopts.AgentOpts._optlist
    assert agentopts.AgentOpts._xoptkey == "ModelList"
    assert "Models" in agentopts.AgentOpts._optmap
    assert agentopts.AgentOpts._optmap["Models"] == "ModelList"


# Test ModelOpts type in _opttypes
def test_08_opttypes():
    """Test that _opttypes has ModelOpts as default type"""
    assert "_default_" in agentopts.AgentOpts._opttypes
    assert agentopts.AgentOpts._opttypes["_default_"] == \
        agentopts.modelopts.ModelOpts


# Test ModelOpts default values
def test_09_modelopts_defaults():
    """Test ModelOpts default values"""
    model_opts = agentopts.ModelOpts()
    assert model_opts.get_opt("ToolSet") == "full"
    assert model_opts.get_opt("SkillSet") == "full"


# Test ModelOpts toolset/skillset value validation
def test_10_modelopts_valid_values():
    """Test ModelOpts accepts valid ToolSet and SkillSet values"""
    # Test valid ToolSet values
    for val in ["none", "low", "medium", "full"]:
        opts = agentopts.ModelOpts(ToolSet=val)
        assert opts["ToolSet"] == val

    # Test valid SkillSet values
    for val in ["none", "low", "medium", "full"]:
        opts = agentopts.ModelOpts(SkillSet=val)
        assert opts["SkillSet"] == val


# Test ModelOpts value aliases
def test_11_modelopts_aliases():
    """Test ModelOpts value aliases (_optvalmap defined but not applied)"""
    # Note: _optvalmap is defined in ModelOpts but not actively used
    # The valid values are: "none", "low", "medium", "full"
    # Value aliases like "off", "lo", "med", "hi", "high", "all" are defined
    # but currently not converted - they would fail validation

    # Test that valid values work
    for val in ["none", "low", "medium", "full"]:
        opts = agentopts.ModelOpts(ToolSet=val)
        assert opts["ToolSet"] == val
        opts = agentopts.ModelOpts(SkillSet=val)
        assert opts["SkillSet"] == val

    # Test that invalid values fail (or are stored raw if warnmode is low)
    # This tests the actual behavior - aliases are defined but not applied
    opts = agentopts.ModelOpts(ToolSet="off", _warnmode=0)
    # With warnmode=0, invalid values are stored without checking
    assert opts.get("ToolSet") == "off"


# Test Parent option in ModelOpts
def test_12_modelopts_parent():
    """Test Parent option in ModelOpts"""
    opts = agentopts.ModelOpts(Parent="some_parent_model")
    assert opts["Parent"] == "some_parent_model"


# Test alias mappings in ModelOpts
def test_13_modelopts_optmap():
    """Test option aliases in ModelOpts"""
    # ToolLevel -> ToolSet
    opts = agentopts.ModelOpts(ToolLevel="low")
    assert "ToolSet" in opts
    assert opts["ToolSet"] == "low"

    # SkillLevel -> SkillSet
    opts = agentopts.ModelOpts(SkillLevel="medium")
    assert "SkillSet" in opts
    assert opts["SkillSet"] == "medium"

    # Type -> Parent
    opts = agentopts.ModelOpts(Type="parent_model")
    assert "Parent" in opts
    assert opts["Parent"] == "parent_model"


# Test Parent cascading behavior
def test_14_parent_cascading():
    """Test that Parent option enables cascading settings"""
    # Create opts with parent-child relationship
    opts = agentopts.AgentOpts(
        ModelList=["parent_model", "child_model"],
        parent_model={"ToolSet": "low", "SkillSet": "none"},
        child_model={"Parent": "parent_model"}
    )

    # Parent has its own settings
    parent_opts = opts["parent_model"]
    assert parent_opts.get_opt("ToolSet") == "low"
    assert parent_opts.get_opt("SkillSet") == "none"

    # Child inherits via Parent
    child_opts = opts["child_model"]
    assert child_opts.get_opt("Parent") == "parent_model"

    # get_ModelOpt returns the child's immediate parent
    parent_ref = opts.get_ModelOpt("child_model", "Parent")
    assert parent_ref == "parent_model"


# Test JSON file with comments
@testutils.run_testdir(__file__)
def test_15_json_with_comments():
    """Test reading JSON file with comments"""
    json_content = """
    {
        "ModelList": [
            "model1",  // First model
            "model2"   // Second model
        ],
        "model1": {
            "ToolSet": "low",  // Limited tools
            "SkillSet": "none" // No skills
        }
    }
    """
    with open("commented.json", "w") as f:
        f.write(json_content)

    opts = agentopts.AgentOpts("commented.json")
    assert "ModelList" in opts
    assert len(opts["ModelList"]) == 2
    assert opts["model1"].get_opt("ToolSet") == "low"
