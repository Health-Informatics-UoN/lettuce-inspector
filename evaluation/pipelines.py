from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from torch.functional import Tensor
from evaluation.evaltypes import SingleResultPipeline, InformationRetrievalPipeline
from jinja2 import Template
from llama_cpp import Llama
from omop.omop_queries import query_vector
from typing import List

from query_handler.handler_type import ConceptIDQueryHandler

class LLMPipeline(SingleResultPipeline):
    """
    This class runs a simple LLM-only pipeline on provided input
    """

    def __init__(
        self, llm: Llama, prompt_template: Template, template_vars: list[str]
    ) -> None:
        """
        Initialises the LLMPipeline class

        Parameters
        ----------
        llm: LLMModel
            One of the model options in the LLMModel enum
        prompt_template: Template
            A jinja2 template for a prompt
        template_vars: list[str]
            The variables inserted into the prompt template when rendered
        """
        self.prompt_template = prompt_template
        self._model = llm
        self._template_vars = template_vars

    def run(self, input: list[str]) -> str:
        """
        Runs the LLMPipeline on a given input

        Parameters
        ----------
        input: list[str]
            The input strings passed to the prompt template, in the order the template_vars were provided to the class

        Returns
        -------
        str
            The output of running the prompt through the given model
        """
        prompt = self.prompt_template.render(
            {(v, i) for v, i in zip(self._template_vars, input)}
        )
        reply = self._model.create_completion(prompt=prompt)["choices"][0]["text"]
        print(f"Replied {reply} for {input}")
        return reply

    def drop(self):
        del self._model


class EmbeddingsPipeline(SingleResultPipeline):

    def __init__(self, embedding_model: SentenceTransformer) -> None:
        self.model = embedding_model

    def run(self, input: str) -> Tensor:
        return self.model.encode(input)

class DBRankRetrievalPipeline(InformationRetrievalPipeline):
    def __init__(self, retriever: ConceptIDQueryHandler) -> None:
        self._retriever = retriever

    def run(self, query: str) -> List[int]:
        return self._retriever.search([query])[0]

class EmbeddingsRetrievalPipeline(InformationRetrievalPipeline):
    def __init__(self, embedding_model: SentenceTransformer, retriever: ConceptIDQueryHandler) -> None:
        self._model = embedding_model
        self._retriever = retriever

    def run(self, query: str) -> List[int]:
        return self._retriever.search([query])[0]

class RAGPipeline(SingleResultPipeline):
    def __init__(
        self,
        llm: Llama,
        prompt_template: Template,
        template_vars: list[str],
        embedding_model: SentenceTransformer,
        session: Session,
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self._llmodel = llm
        self._embedding_model = embedding_model
        self._template_vars = template_vars
        self._top_k = top_k
        self._session = session

    def run(self, input: list[str]) -> str:
        embedding = self._embedding_model.encode(input[0])
        # In future, this could be generalised
        # We could create a query_handler that fetches from a pgvector enabled database and RAGPipelines could use a generic query_handler
        search_query = query_vector(embedding, self._top_k)
        search_results = {
            "documents": self._session.execute(search_query).mappings().all()
        }
        prompt = self.prompt_template.render(
            dict(zip(self._template_vars, [*input, search_results["documents"]]))
        )
        reply = self._llmodel.create_completion(prompt=prompt)["choices"][0]["text"]
        print(f"Replied {reply} for {input}")
        return reply

class AugmentedQueryPipeline(InformationRetrievalPipeline):
    def __init__(
            self,
            prompt_template: Template,
            llm: Llama,
            template_vars: List[str],
            retriever: ConceptIDQueryHandler,
            ) -> None:
        self._prompt_template = prompt_template
        self._llmodel = llm
        self._template_vars = template_vars
        self._retriever = retriever

    def run(self, input: List[str]) -> List[int]:
        prompt = self._prompt_template.render(
                {"informal_name": input}
                )
        print(prompt)
        reply = self._llmodel.create_chat_completion(
            messages = [
                    {
                        "role": "system",
                        "content": """You are an assistant that suggests terms for semantic search.
Respond only with a suggestion similar to a standardised term for the informal name, without any extra explanation. For example, if given the name of the medication, give your best guess for the medication's formal name.
If you are given a formal name, just echo it"""},
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )["choices"][0]["message"]["content"]
        print(reply)
        return self._retriever.search([reply])[0]
