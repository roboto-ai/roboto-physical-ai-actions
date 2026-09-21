import inspect
import types
import typing

import roboto
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
