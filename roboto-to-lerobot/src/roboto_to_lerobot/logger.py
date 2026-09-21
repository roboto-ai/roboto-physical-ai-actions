import logging

logging.basicConfig(
    format="[%(levelname)4s:%(filename)s %(lineno)4s] %(message)s",
)
logger = logging.getLogger(name="roboto_to_lerobot")
