import inspect
import types
import typing

import pytest
import roboto
import roboto_to_lerobot.main as main_mod
from roboto_to_lerobot.main import main


def test_main_function_signature_matches_expectation():
    """
    Validates that the main function conforms to the interface requirements
    documented in DEVELOPING.md:
    - Function signature: def main(context: roboto.InvocationContext) -> None
    - Single parameter of type roboto.InvocationContext
    - Return type of None
    """
    # Arrange / Act
    signature = inspect.signature(main)
    type_hints = typing.get_type_hints(main)
    parameters = list(signature.parameters.values())

    # Assert
    assert len(parameters) == 1, (
        f"main() must accept exactly {1} parameter, but found {len(parameters)}"
    )

    param = parameters[0]
    assert param.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    }, (
        f"main() parameter 'context' must be positional or positional-or-keyword, "
        f"but found {param.kind}"
    )

    if param.name in type_hints:
        assert type_hints[param.name] == roboto.InvocationContext, (
            f"single main() parameter expected to be type of {roboto.InvocationContext.__name__}"
        )

    if "return" in type_hints:
        assert type_hints["return"] is types.NoneType, "main() expected to return None"


def _video_context(**params):
    return types.SimpleNamespace(get_optional_parameter=params.get)


def test_video_parameters_map_to_encoder_settings():
    """The set video parameters come back keyed by encoder setting; unset ones are left out."""
    overrides = main_mod._parse_video_overrides(_video_context(video_crf="20", video_gop=8))
    assert overrides == {"crf": 20, "g": 8}


@pytest.mark.parametrize(
    ("name", "value"),
    [("video_crf", "0"), ("video_crf", "64"), ("video_crf", "high"), ("video_preset", "14"), ("video_gop", "0")],
)
def test_video_parameter_out_of_range_fails(name, value):
    """A video parameter that is not an integer in its range fails the run."""
    with pytest.raises(ValueError, match=f"{name} must be an integer"):
        main_mod._parse_video_overrides(_video_context(**{name: value}))
