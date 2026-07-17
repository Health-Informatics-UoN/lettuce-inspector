import sys
import logging
import duckdb

from evaluation.eval_data_loaders import SingleInputCSVforLLM
from evaluation.evaltypes import EvaluationFramework
from evaluation.metrics import (
    IDMatchInList,
    IDMatchPositionInList,
)
from evaluation.pipelines import DBRankRetrievalPipeline
from evaluation.evaltypes import InformationRetrievalPipelineTest
from omop.db_manager import db_session
from query_handler.duckdb_connectors import connect_to_concept_csv
from query_handler.text_query import BmFileSearcher

logger = logging.Logger("BM25-test")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

description = """
Testing BM25 for the Lettuce paper

Dataset: 400 HELIOS self-reported medications

## Pipelines
Retrieval: BM25, top 20

## Metrics
Retrieval: IDMatchInList, IDMatchPositionInList
"""

logger.info("Connecting to evaluation dataset")
dataset = SingleInputCSVforLLM("evaluation/datasets/helios-eval-set.csv")

logger.info("Connecting to evaluation database")
eval_conn = db_session()

logger.info("Initialising retrieval database")
db = duckdb.connect()
connect_to_concept_csv(db, "~/omop-lite/data/CONCEPT.csv")

logger.info("Loading retriever")
bm25 = BmFileSearcher(db=db, vocabulary_ids=None, top_k=20)
bm25_pipeline = DBRankRetrievalPipeline(retriever=bm25)
logger.info("Retriever initialised")


retrieval_metrics = [
    IDMatchInList(
        connection=eval_conn, vocabulary_ids=["RxNorm", "RxNorm Extension", "SNOMED"]
    ),
    IDMatchPositionInList(
        connection=eval_conn, vocabulary_ids=["RxNorm", "RxNorm Extension", "SNOMED"]
    ),
]

tests = [
    InformationRetrievalPipelineTest(
        "BM25 retrieval", bm25_pipeline, retrieval_metrics
    ),
]

framework = EvaluationFramework(
    name="BM25 test",
    pipeline_tests=tests,
    dataset=dataset,
    description=description,
    results_path="paper-scripts/results.json",
)

if __name__ == "__main__":
    framework.run_evaluations()
