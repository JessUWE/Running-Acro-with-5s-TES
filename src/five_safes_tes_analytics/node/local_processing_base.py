from abc import ABC, abstractmethod
from sqlalchemy.engine import Result, Engine
import pathlib
from typing import Optional

class BaseLocalProcessing(ABC):
    """
    Abstract base class for local processing operations in federated analysis.
    
    Each subclass represents a different analysis type that can be run on individual
    TREs. The class handles SQL query building and optional Python-side analysis,
    returning results that can be aggregated across multiple TREs.
    """
    analysis_type: str | None
    user_query: str
    engine: Engine
    output_folder: pathlib.Path

    def __init__(
        self,
        analysis_type: str | None = None,
        user_query: str | None = None,
        engine: Engine | None = None,
        output_folder: pathlib.Path = pathlib.Path("/outputs"),
    ):
        # Use class attribute as default if no analysis_type provided
        self.analysis_type = analysis_type if analysis_type is not None else getattr(self.__class__, "analysis_type", None)

        if user_query is None:
            raise ValueError("User query must be provided")
        self.user_query = user_query

        if engine is None:
            raise ValueError("Engine must be provided")
        self.engine = engine

        self.output_folder = output_folder

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the processing step."""
        raise NotImplementedError("Description must be implemented in subclass")

    @property
    def processing_query(self) -> str|None:
        """SQL fragment for the processing step. By default, returns None."""
        return None

    @property
    @abstractmethod
    def user_query_requirements(self) -> str:
        """Requirements for the user query."""
        raise NotImplementedError("User query requirements must be implemented in subclass")

    def build_query(self) -> str:
        """
        Build a complete SQL query by combining user's data selection with analysis calculations.
        """
        if self.processing_query is None:
            return self.user_query
        
        # Combine user query with analysis part
        query = f"""WITH user_query AS (
{self.user_query}
)
{self.processing_query}"""
        return query

    def python_analysis(self, sql_result: Result) -> dict | list[dict] | None:
        """
        Optional Python-side analysis. Override in subclasses if needed.
        By default, does nothing and returns None.
        
        Args:
            sql_result: SQLAlchemy Result object from query execution
            
        Returns:
            dict: For single result objects (e.g., {"total": 100, "n": 10})
            list[dict]: For multiple rows (e.g., [{"gender": "Male", "n": 5}, {"gender": "Female", "n": 3}])
            None: If no Python analysis is needed (SQL results will be used instead)
            
        Examples:
            Single dict: {"centroids": [...], "count": 100}  # PercentileSketch
            List of dicts: [{"category": "A", "n": 10}, {"category": "B", "n": 20}]  # Contingency tables
        """
        return None
