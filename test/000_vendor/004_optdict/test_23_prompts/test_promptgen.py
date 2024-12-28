
# Local imports
from cape.optdict import OptionsDict


# Custom subsection class
class MySubsubsec(OptionsDict):
    _name = "custom sub-subsection"

    _optlist = (
        "aa",
        "bb",
    )


class MySubsec(OptionsDict):
    _name = "custom subsection"

    _optlist = (
        "a",
        "b",
        "s",
    )

    _sec_cls = {
        "s": MySubsubsec,
    }

    _rst_descriptions = {
        "a": "angle setting",
        "b": "bluntness setting",
    }


# Main custom options
class MyOpts(OptionsDict):
    _optlist = (
        "sub",
        "c",
    )

    _rst_descriptions = {
        "c": "cubicle setting",
    }

    _sec_cls = {
        "sub": MySubsec,
    }


# Test prompt generation
def test_promptgen():
    # Instantiate
    opts = MyOpts(
        {
            "c": 3.0,
            "sub": {
                "a": 15.1,
                "b": "really",
                "s": {
                    "aa": 1,
                    "bb": 2,
                }
            }
        })
    # Create prompts
    prompts = opts.genr8_prompts()
    # Test output
    assert isinstance(prompts, list)
    assert isinstance(prompts[0], tuple)
    assert len(prompts[0]) == 2
