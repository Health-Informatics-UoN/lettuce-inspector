from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from torch.functional import Tensor
from evaluation.evaltypes import SingleResultPipeline, InformationRetrievalPipeline
from jinja2 import Template
from llama_cpp import Llama
from omop.omop_queries import query_vector
from typing import Callable, final

from query_handler.duckdb_queries import bm25_query, vector_search
from query_handler.handler_type import ConceptIDQueryHandler
from query_handler.vectors import SimilarityFunction
import duckdb

@final
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

@final
class LLMChatPipeline(SingleResultPipeline):
    """
    This class runs a simple LLM-only pipeline on provided input
    """

    def __init__(
            self, llm: Llama,
            system_prompt:str,
            prompt_template: Template,
            template_vars: list[str],
            verbose: bool = False,
    ) -> None:
        """
        Initialises the LLMPipeline class

        Parameters
        ----------
        llm: LLMModel
            One of the model options in the LLMModel enum
        system_prompt: str
            A system prompt for the chat model
        prompt_template: Template
            A jinja2 template for a prompt
        template_vars: list[str]
            The variables inserted into the prompt template when rendered
        """
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self._model = llm
        self._template_vars = template_vars
        self._verbose = verbose

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
        if self._verbose:
            print(prompt)
        reply = self._model.create_chat_completion(
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                    ]
                )["choices"][0]["message"]["content"]
        if self._verbose:
            print(f"Replied {reply} for {input}")
        return reply

    def drop(self):
        del self._model


class EmbeddingsPipeline(SingleResultPipeline):

    def __init__(self, embedding_model: SentenceTransformer) -> None:
        self.model: SentenceTransformer = embedding_model

    def run(self, input: str) -> Tensor:
        return self.model.encode(input)

class DBRankRetrievalPipeline(InformationRetrievalPipeline):
    def __init__(self, retriever: ConceptIDQueryHandler) -> None:
        self._retriever: ConceptIDQueryHandler = retriever

    def run(self, query: str) -> list[int]:
        return self._retriever.search([query])[0]

class EmbeddingsRetrievalPipeline(InformationRetrievalPipeline):
    def __init__(self, embedding_model: SentenceTransformer, retriever: ConceptIDQueryHandler) -> None:
        self._model: SentenceTransformer = embedding_model
        self._retriever: ConceptIDQueryHandler = retriever

    def run(self, query: str) -> list[int]:
        return self._retriever.search([query])[0]

@final
class RAGPipeline(SingleResultPipeline):
    def __init__(
        self,
        llm: Llama,
        prompt_template: Template,
        template_vars: list[str],
        embedding_model: SentenceTransformer,
        session: Session,
        embed_vocab: list[str] | None = None,
        domain_id: list[str] | None = None,
        standard_concept: bool = False,
        top_k: int = 5,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self._llmodel = llm
        self._embedding_model = embedding_model
        self._template_vars = template_vars
        self._embed_vocab = embed_vocab
        self._domain_id = domain_id
        self._standard_concept = standard_concept
        self._top_k = top_k
        self._session = session
        self._verbose = verbose

    def run(self, input: list[str]) -> str:
        embedding = self._embedding_model.encode(input[0])
        # In future, this could be generalised
        # We could create a query_handler that fetches from a pgvector enabled database and RAGPipelines could use a generic query_handler
        search_query = query_vector(embedding,
                                    embed_vocab= self._embed_vocab,
                                    domain_id= self._domain_id,
                                    standard_concept= self._standard_concept,
                                    n=self._top_k)
        search_results = {
            "documents": self._session.execute(search_query).mappings().all()
        }
        prompt = self.prompt_template.render(
            dict(zip(self._template_vars, [*input, search_results["documents"]]))
        )
        if self._verbose:
            print(prompt)
        reply = self._llmodel.create_completion(prompt=prompt)["choices"][0]["text"]
        print(f"Replied {reply} for {input}")
        return reply

@final
class RAGChatPipeline(SingleResultPipeline):
    def __init__(
        self,
        llm: Llama,
        system_prompt: str,
        prompt_template: Template,
        template_vars: list[str],
        embedding_model: SentenceTransformer,
        session: Session,
        embed_vocab: list[str] | None = None,
        domain_id: list[str] | None = None,
        standard_concept: bool = False,
        top_k: int = 5,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self._llmodel = llm
        self._embedding_model = embedding_model
        self._template_vars = template_vars
        self._embed_vocab = embed_vocab
        self._domain_id = domain_id
        self._standard_concept = standard_concept
        self._top_k = top_k
        self._session = session
        self._verbose = verbose

    def run(self, input: list[str]) -> str:
        embedding = self._embedding_model.encode(input[0])
        # We could create a query_handler that fetches from a pgvector enabled database and RAGPipelines could use a generic query_handler
        search_query = query_vector(embedding,
                                    embed_vocab= self._embed_vocab,
                                    domain_id= self._domain_id,
                                    standard_concept= self._standard_concept,
                                    n=self._top_k)
        search_results = {
            "documents": self._session.execute(search_query).mappings().all()
        }
        prompt = self.prompt_template.render(
            dict(zip(self._template_vars, [*input, search_results["documents"]]))
        )
        if self._verbose:
            print(prompt)
        reply = self._llmodel.create_chat_completion(
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                    ]
                )["choices"][0]["message"]["content"]
        if self._verbose:
            print(f"Replied {reply} for {input}")
        return reply

@final
class BM25RAGChatPipeline(SingleResultPipeline):
    def __init__(
        self,
        llm: Llama,
        system_prompt: str,
        prompt_template: Template,
        template_vars: list[str],
        connection: duckdb.DuckDBPyConnection,
        domain_id: list[str] | None = None,
        standard_concept: bool = False,
        top_k: int = 5,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self._llmodel = llm
        self._template_vars = template_vars
        self._domain_id = domain_id
        self._standard_concept = standard_concept
        self._top_k = top_k
        self._verbose = verbose
        self._connection = connection

    def run(self, input: list[str]) -> str:
        bm25 = bm25_query(
                con=self._connection,
                query=input[0],
                vocabulary_ids=None,
                top_k=self._top_k,
                ).fetchall()

        prompt = self.prompt_template.render(
            dict(zip(self._template_vars, [*input, bm25]))
        )
        if self._verbose:
            print(prompt)
        reply = self._llmodel.create_chat_completion(
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                    ]
                )["choices"][0]["message"]["content"]
        if self._verbose:
            print(f"Replied {reply} for {input}")
        return reply

@final
class DuckdbRAGChatPipeline(SingleResultPipeline):
    def __init__(
        self,
        llm: Llama,
        system_prompt: str,
        prompt_template: Template,
        template_vars: list[str],
        embedding_model: SentenceTransformer,
        db: duckdb.DuckDBPyConnection,
        similarity_function: SimilarityFunction = SimilarityFunction.COSINE_DISTANCE,
        vocabulary_ids: list[str] | None = None,
        top_k: int = 5,
        vector_dimension: int = 384,
        vector_type: str = "DOUBLE",
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self._llmodel = llm
        self._embedding_model = embedding_model
        self._template_vars = template_vars
        self._db = db
        self._sim = similarity_function
        self._vocabulary_ids = vocabulary_ids
        self._top_k = top_k
        self._vector_dim = vector_dimension
        self._vector_type = vector_type
        self._verbose = verbose

    def run(self, input: list[str]) -> str:
        # Generate embedding for the query
        embedding = self._embedding_model.encode(input[0])
        if len(embedding.shape) > 1:
            embedding = embedding[0]
        
        # Perform vector search using DuckDB
        search_results_raw = vector_search(
            con=self._db,
            similarity_function=self._sim,
            embedding=embedding,
            vector_type=self._vector_type,
            vector_dim=self._vector_dim,
            vocabulary_ids=self._vocabulary_ids,
            top_k=self._top_k,
        ).fetchall()
        
        # Format results to match the expected structure (similar to SQLAlchemy mappings)
        search_results = {
            "documents": [
                {
                    "concept_id": row[1],
                    "score": row[2],
                    "domain": row[3],
                    "vocabulary": row[4],
                    "concept_class": row[5],
                    }
                for row in search_results_raw
            ]
        }
        
        # Render the prompt with input and search results
        prompt = self.prompt_template.render(
            dict(zip(self._template_vars, [*input, search_results["documents"]]))
        )
        
        if self._verbose:
            print(prompt)
        
        # Generate response using LLM
        reply = self._llmodel.create_chat_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )["choices"][0]["message"]["content"]
        
        if self._verbose:
            print(f"Replied {reply} for {input}")
        
        return reply

class AugmentedQueryPipeline(InformationRetrievalPipeline):
    def __init__(
            self,
            prompt_template: Template,
            llm: Llama,
            template_vars: list[str],
            retriever: ConceptIDQueryHandler,
            ) -> None:
        self._prompt_template = prompt_template
        self._llmodel = llm
        self._template_vars = template_vars
        self._retriever = retriever

    def run(self, input: list[str]) -> list[int]:
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
