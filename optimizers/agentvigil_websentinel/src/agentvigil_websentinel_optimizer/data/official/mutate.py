import logging
import textwrap
from collections.abc import Iterable

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from mutate_prompts import (
    MutationMethod,
    mutation_prompt_templates,
    postprocess_mutated_text,
    system_prompt,
)

logger = logging.getLogger(__name__)


class Mutator:
    def __init__(
        self,
        helper_model: str = "gpt-4o-mini-2024-07-18",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
    ):
        self.helper_client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, max_retries=max_retries
        )
        self.helper_model_id = helper_model
        logger.info(f"Initialized Mutator with model={helper_model}, max_retries={max_retries}")

    async def mutate(
        self, seed_text: str | list[str], method: str | MutationMethod
    ) -> str | None:
        method = MutationMethod(method)
        logger.debug(f"Starting mutation with method={method}")
        if method == MutationMethod.Crossover:
            if not isinstance(seed_text, list) or len(seed_text) != 2:
                logger.error("Crossover mutation requires two seed texts")
                raise ValueError("Crossover mutation requires two seed texts")
            mutated_text = await self.crossover_mutate(seed_text[0], seed_text[1])
        else:
            if not isinstance(seed_text, str):
                logger.error("Simple mutation requires a single seed text")
                raise ValueError("Simple mutation requires a single seed text")
            mutated_text = await self.simple_mutate(seed_text, method)

        if mutated_text is not None:
            mutated_text = postprocess_mutated_text(mutated_text)
            logger.debug(f"Mutation successful with method={method}")
        else:
            logger.warning(f"Mutation failed with method={method}")

        return mutated_text

    async def query_helper_model(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> str | None:
        try:
            logger.debug(f"Querying helper model: {self.helper_model_id}")
            response = await self.helper_client.chat.completions.create(
                model=self.helper_model_id,
                messages=messages,
            )
            logger.debug("Successfully received response from helper model")
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Failed to query helper model: {e}")
            return None

    @staticmethod
    def show_response(response: str | None) -> None:
        if response:
            formatted_response = "\n".join(textwrap.wrap(response, width=80))
            formatted_response = textwrap.indent(formatted_response, "\t")
            logger.debug(f"Response from helper model:\n{formatted_response}")

    async def simple_mutate(self, seed_text: str, method: MutationMethod) -> str | None:
        logger.debug(f"Performing simple mutation with method={method}")
        messages = [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
            ChatCompletionUserMessageParam(
                role="user",
                content=mutation_prompt_templates[method]
                % (seed_text, "the key placeholders"),
            ),
        ]

        response = await self.query_helper_model(messages)

        self.show_response(response)

        return response

    async def crossover_mutate(self, seed1: str, seed2: str) -> str | None:
        logger.debug("Performing crossover mutation")
        messages = [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
            ChatCompletionUserMessageParam(
                role="user",
                content=mutation_prompt_templates[MutationMethod.Crossover]
                % (seed1, seed2, "the key placeholders"),
            ),
        ]

        response = await self.query_helper_model(messages)

        self.show_response(response)

        return response
