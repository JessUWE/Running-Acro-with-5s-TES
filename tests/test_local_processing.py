import pytest
import json
import os
from unittest.mock import Mock, MagicMock, patch
from sqlalchemy import text
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from five_safes_tes_analytics.node.local_processing import (
    BaseLocalProcessing, Mean, Variance, PMCC, ContingencyTable, PercentileSketch,
    AcroTableMeans, AcroTableCounts,
    LOCAL_PROCESSING_CLASSES, get_local_processing_registry
)


def create_mock_engine_with_connection():
    """Helper function to create properly mocked database engine and connection."""
    mock_engine = Mock()
    mock_conn = Mock()
    
    # Set up context manager properly
    mock_connection_context = Mock()
    mock_connection_context.__enter__ = Mock(return_value=mock_conn)
    mock_connection_context.__exit__ = Mock(return_value=None)
    
    mock_engine.connect.return_value = mock_connection_context
    return mock_engine, mock_conn


class TestBaseLocalProcessing:
    """Test the base local processing class."""
    
    def test_base_class_abstract_methods(self):
        """Test that BaseLocalProcessing is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseLocalProcessing()
    
    def test_build_query_without_analysis_type(self):
        """Test build_query when analysis_type is None raises ValueError."""
        class TestProcessor(BaseLocalProcessing):
            @property
            def description(self):
                return "Test"
            
            @property
            def processing_query(self):
                return "SELECT * FROM test"
            
            @property
            def user_query_requirements(self):
                return "Test requirements"
        
        processor = TestProcessor(analysis_type=None, user_query="SELECT * FROM users")
        with pytest.raises(ValueError, match="Unsupported analysis type: None"):
            processor.build_query()
    
    def test_build_query_with_unsupported_analysis_type(self):
        """Test build_query with unsupported analysis type."""
        class TestProcessor(BaseLocalProcessing):
            @property
            def description(self):
                return "Test"
            
            @property
            def processing_query(self):
                return "SELECT * FROM test"
            
            @property
            def user_query_requirements(self):
                return "Test requirements"
        
        processor = TestProcessor(analysis_type="unsupported", user_query="SELECT * FROM users")
        with pytest.raises(ValueError, match="Unsupported analysis type"):
            processor.build_query()
    
    def test_build_query_with_valid_analysis(self):
        """Test build_query with valid analysis type."""
        class TestProcessor(BaseLocalProcessing):
            analysis_type = "mean"
            
            @property
            def description(self):
                return "Test"
            
            @property
            def processing_query(self):
                return "SELECT COUNT(*) FROM user_query"
            
            @property
            def user_query_requirements(self):
                return "Test requirements"
        
        processor = TestProcessor(analysis_type="mean", user_query="SELECT * FROM users")
        query = processor.build_query()
        expected = """WITH user_query AS (
SELECT * FROM users
)
SELECT COUNT(*) FROM user_query"""
        assert query == expected
    
    def test_python_analysis_default(self):
        """Test default python_analysis method returns None."""
        class TestProcessor(BaseLocalProcessing):
            @property
            def description(self):
                return "Test"
            
            @property
            def processing_query(self):
                return "SELECT * FROM test"
            
            @property
            def user_query_requirements(self):
                return "Test requirements"
        
        processor = TestProcessor()
        result = processor.python_analysis("test_data")
        assert result is None


class TestMean:
    """Test the Mean processing class."""
    
    def test_mean_properties(self):
        """Test Mean class properties."""
        processor = Mean()
        assert processor.analysis_type == "mean"
        assert processor.description == "Calculate mean of a numeric column"
        assert processor.user_query_requirements == "Must select a single numeric column"
    
    def test_mean_processing_query(self):
        """Test Mean processing query structure."""
        processor = Mean()
        query = processor.processing_query
        assert "COUNT(*)" in query
        assert "SUM(value_as_number)" in query
        assert "AS n" in query
        assert "AS total" in query
    
    def test_mean_build_query(self):
        """Test Mean build_query method."""
        user_query = "SELECT value_as_number FROM measurements WHERE value_as_number IS NOT NULL"
        processor = Mean(user_query=user_query)
        query = processor.build_query()
        
        assert "WITH user_query AS" in query
        assert user_query in query
        assert "COUNT(*) AS n" in query
        assert "SUM(value_as_number) AS total" in query


class TestVariance:
    """Test the Variance processing class."""
    
    def test_variance_properties(self):
        """Test Variance class properties."""
        processor = Variance()
        assert processor.analysis_type == "variance"
        assert processor.description == "Calculate variance of a numeric column"
        assert processor.user_query_requirements == "Must select a single numeric column"
    
    def test_variance_processing_query(self):
        """Test Variance processing query structure."""
        processor = Variance()
        query = processor.processing_query
        assert "COUNT(*)" in query
        assert "SUM(value_as_number * value_as_number)" in query
        assert "SUM(value_as_number)" in query
        assert "AS sum_x2" in query


class TestPMCC:
    """Test the PMCC processing class."""
    
    def test_pmcc_properties(self):
        """Test PMCC class properties."""
        processor = PMCC()
        assert processor.analysis_type == "PMCC"
        assert processor.description == "Calculate Pearson's correlation coefficient between two numeric columns"
        assert processor.user_query_requirements == "Must select exactly two numeric columns (x and y)"
    
    def test_pmcc_processing_query(self):
        """Test PMCC processing query structure."""
        processor = PMCC()
        query = processor.processing_query
        assert "COUNT(*)" in query
        assert "SUM(x)" in query
        assert "SUM(y)" in query
        assert "SUM(x * x)" in query
        assert "SUM(y * y)" in query
        assert "SUM(x * y)" in query


class TestContingencyTable:
    """Test the ContingencyTable processing class."""
    
    def test_contingency_table_properties(self):
        """Test ContingencyTable class properties."""
        processor = ContingencyTable()
        assert processor.analysis_type == "contingency_table"
        assert processor.description == "Build a contingency table from one or more categorical columns"
        assert processor.user_query_requirements == "Must select one or more categorical columns"
    
    def test_get_columns_from_user_query_with_limit(self):
        """Test get_columns_from_user_query when query already has LIMIT."""
        mock_engine, mock_conn = create_mock_engine_with_connection()
        mock_result = Mock()
        mock_result.keys.return_value = ["gender", "race"]
        
        mock_conn.execute.return_value = mock_result
        
        processor = ContingencyTable(engine=mock_engine, user_query="SELECT gender, race FROM patients LIMIT 10")
        columns = processor.get_columns_from_user_query()
        
        assert columns == ["gender", "race"]
        mock_conn.execute.assert_called_once()
    
    def test_get_columns_from_user_query_without_limit(self):
        """Test get_columns_from_user_query when query doesn't have LIMIT."""
        mock_engine = MagicMock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.keys.return_value = ["gender", "race"]
        
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value = mock_result
        
        processor = ContingencyTable(engine=mock_engine, user_query="SELECT gender, race FROM patients")
        columns = processor.get_columns_from_user_query()
        
        assert columns == ["gender", "race"]
        # Should add LIMIT 0 to the query
        call_args = mock_conn.execute.call_args[0][0]
        assert "LIMIT 0" in str(call_args)
    
    def test_get_columns_from_user_query_no_columns(self):
        """Test get_columns_from_user_query when no columns are found."""
        mock_engine = MagicMock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.keys.return_value = []
        
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value = mock_result
        
        processor = ContingencyTable(engine=mock_engine, user_query="SELECT * FROM empty_table")
        
        with pytest.raises(ValueError, match="No columns found in user query"):
            processor.get_columns_from_user_query()
    
    def test_contingency_table_processing_query(self):
        """Test ContingencyTable processing query structure."""
        mock_engine = MagicMock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.keys.return_value = ["gender", "race"]
        
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value = mock_result
        
        processor = ContingencyTable(engine=mock_engine, user_query="SELECT gender, race FROM patients")
        query = processor.processing_query
        
        assert "SELECT" in query
        assert "gender, race" in query
        assert "COUNT(*) AS n" in query
        assert "GROUP BY gender, race" in query
        assert "ORDER BY gender, race" in query


class TestPercentileSketch:
    """Test the PercentileSketch processing class."""
    
    def test_percentile_sketch_properties(self):
        """Test PercentileSketch class properties."""
        processor = PercentileSketch()
        assert processor.analysis_type == "percentile_sketch"
        assert processor.description == "Calculate percentile sketch of a numeric column"
        assert processor.user_query_requirements == "Must select a numeric column"
        assert processor.processing_query is None
    
    def test_percentile_sketch_python_analysis(self):
        """Test PercentileSketch python_analysis method."""
        processor = PercentileSketch()
        
        # Mock SQL result with some values
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            (10.5,),
            (20.3,),
            (None,),  # Should be filtered out
            (15.7,),
        ]
        
        result = processor.python_analysis(mock_result)
        
        # Should return JSON string
        assert isinstance(result, str)
        json_data = json.loads(result)
        assert "centroids" in json_data
        assert "n" in json_data


class TestRegistry:
    """Test the processing class registry."""
    
    def test_get_local_processing_registry(self):
        """Test that registry contains all expected classes."""
        registry = get_local_processing_registry()
        
        expected_types = ["mean", "variance", "PMCC", "contingency_table", "percentile_sketch"]
        for analysis_type in expected_types:
            assert analysis_type in registry
    
    def test_local_processing_classes_constant(self):
        """Test that LOCAL_PROCESSING_CLASSES constant is properly set."""
        assert "mean" in LOCAL_PROCESSING_CLASSES
        assert "variance" in LOCAL_PROCESSING_CLASSES
        assert "PMCC" in LOCAL_PROCESSING_CLASSES
        assert "contingency_table" in LOCAL_PROCESSING_CLASSES
        assert "percentile_sketch" in LOCAL_PROCESSING_CLASSES
    
    def test_registry_class_instantiation(self):
        """Test that classes in registry can be instantiated."""
        registry = LOCAL_PROCESSING_CLASSES
        
        # Test Mean
        mean_processor = registry["mean"]()
        assert isinstance(mean_processor, Mean)
        
        # Test Variance
        variance_processor = registry["variance"]()
        assert isinstance(variance_processor, Variance)
        
        # Test PMCC
        pmcc_processor = registry["PMCC"]()
        assert isinstance(pmcc_processor, PMCC)
        
        # Test ContingencyTable
        contingency_processor = registry["contingency_table"]()
        assert isinstance(contingency_processor, ContingencyTable)
        
        # Test PercentileSketch
        percentile_processor = registry["percentile_sketch"]()
        assert isinstance(percentile_processor, PercentileSketch)


class TestDockerIntegration:
    """Test Docker container integration scenarios."""
    
    def test_mean_analysis_integration(self):
        """Test complete Mean analysis workflow."""
        user_query = "SELECT value_as_number FROM measurements WHERE value_as_number IS NOT NULL"
        
        # Mock database connection
        mock_engine = MagicMock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.keys.return_value = ["n", "total"]
        mock_result.fetchall.return_value = [(100, 1500.5)]
        
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value = mock_result
        
        # Create processor and build query
        processor = Mean(user_query=user_query, engine=mock_engine)
        query = processor.build_query()
        
        # Verify query structure
        assert "WITH user_query AS" in query
        assert user_query in query
        assert "COUNT(*) AS n" in query
        assert "SUM(value_as_number) AS total" in query
        
        # Execute query
        mock_conn.execute(text(query))
        
        # Verify execution was called
        mock_conn.execute.assert_called()
    
    def test_contingency_table_integration(self):
        """Test complete ContingencyTable analysis workflow."""
        user_query = "SELECT gender, race FROM patients"
        
        # Mock database connection for column detection
        mock_engine = MagicMock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.keys.return_value = ["gender", "race"]
        mock_result.fetchall.return_value = [
            ("Male", "White", 45),
            ("Male", "Black", 23),
            ("Female", "White", 52),
            ("Female", "Black", 28)
        ]
        
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value = mock_result
        
        # Create processor
        processor = ContingencyTable(user_query=user_query, engine=mock_engine)
        
        # Get columns
        columns = processor.get_columns_from_user_query()
        assert columns == ["gender", "race"]
        
        # Build query
        query = processor.build_query()
        
        # Verify query structure
        assert "WITH user_query AS" in query
        assert user_query in query
        assert "SELECT" in query
        assert "gender, race" in query
        assert "COUNT(*) AS n" in query
        assert "GROUP BY gender, race" in query
    
    def test_percentile_sketch_integration(self):
        """Test complete PercentileSketch analysis workflow."""
        user_query = "SELECT value_as_number FROM measurements"
        
        # Mock database connection
        mock_engine = MagicMock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.keys.return_value = ["value_as_number"]
        mock_result.fetchall.return_value = [
            (10.5,),
            (20.3,),
            (15.7,),
            (None,),  # Should be filtered out
        ]
        
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value = mock_result
        
        # Create processor
        processor = PercentileSketch(user_query=user_query, engine=mock_engine)
        
        # Build query (should be None for percentile sketch)
        query = processor.build_query()
        assert query == user_query
        
        # Test python analysis
        result = processor.python_analysis(mock_result)
        assert isinstance(result, str)
        
        # Parse JSON result
        json_data = json.loads(result)
        assert "centroids" in json_data
        assert "n" in json_data


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_unsupported_analysis_type(self):
        """Test error handling for unsupported analysis type."""
        class TestProcessor(BaseLocalProcessing):
            @property
            def description(self):
                return "Test"
            
            @property
            def processing_query(self):
                return "SELECT * FROM test"
            
            @property
            def user_query_requirements(self):
                return "Test requirements"
        
        processor = TestProcessor(analysis_type="unsupported", user_query="SELECT * FROM users")
        
        with pytest.raises(ValueError, match="Unsupported analysis type"):
            processor.build_query()
    
    def test_database_connection_error(self):
        """Test error handling for database connection issues."""
        mock_engine = Mock()
        mock_engine.connect.side_effect = Exception("Connection failed")
        
        processor = ContingencyTable(engine=mock_engine, user_query="SELECT * FROM users")
        
        with pytest.raises(Exception, match="Connection failed"):
            processor.get_columns_from_user_query()
    
    def test_empty_query_result(self):
        """Test handling of empty query results."""
        mock_engine = MagicMock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.keys.return_value = []
        
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value = mock_result
        
        processor = ContingencyTable(engine=mock_engine, user_query="SELECT * FROM empty_table")
        
        with pytest.raises(ValueError, match="No columns found in user query"):
            processor.get_columns_from_user_query()


class TestAcroTableMeans:
    """Test the AcroTableMeans processing class."""

    def test_properties(self):
        processor = AcroTableMeans()
        assert processor.analysis_type == "acro_crosstab_mean"
        assert "mean" in processor.description.lower()
        assert processor.processing_query is None
        assert "3 columns" in processor.user_query_requirements

    def test_python_analysis(self):
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("not_recom", "convenient", 3.1),
            ("priority", "inconv", 2.9),
            ("very_recom", "convenient", 3.3),
        ]
        mock_result.keys.return_value = ["recommendation", "finance", "children"]

        mock_acro_instance = Mock()
        mock_acro_instance.finalise.return_value = None
        mock_acro_instance.crosstab.return_value.to_dict.return_value = {"col_a": {"row_1": 3.1}}

        with patch("five_safes_tes_analytics.node.local_processing.acro_module.ACRO", return_value=mock_acro_instance), \
             patch("five_safes_tes_analytics.node.local_processing.shutil.make_archive", return_value="/tmp/acro_output.zip"):
            processor = AcroTableMeans()
            result = processor.python_analysis(mock_result)

        assert isinstance(result, dict)
        assert "acro_output_zip" in result
        assert result["acro_status"] == "finalised"
        assert "table" in result
        mock_acro_instance.crosstab.assert_called_once()
        mock_acro_instance.finalise.assert_called_once()

    def test_crosstab_called_with_mean_aggfunc(self):
        mock_result = Mock()
        mock_result.fetchall.return_value = [("A", "X", 1.0), ("B", "Y", 2.0)]
        mock_result.keys.return_value = ["idx", "col", "val"]

        mock_acro_instance = Mock()

        with patch("five_safes_tes_analytics.node.local_processing.acro_module.ACRO", return_value=mock_acro_instance), \
             patch("five_safes_tes_analytics.node.local_processing.shutil.make_archive", return_value="x.zip"):
            AcroTableMeans().python_analysis(mock_result)

        call_kwargs = mock_acro_instance.crosstab.call_args
        assert call_kwargs.kwargs.get("aggfunc") == "mean"
        assert call_kwargs.kwargs.get("margins") is True

    def test_registered_in_registry(self):
        registry = get_local_processing_registry()
        assert "acro_crosstab_mean" in registry
        assert registry["acro_crosstab_mean"] is AcroTableMeans


class TestAcroTableCounts:
    """Test the AcroTableCounts processing class."""

    def test_properties(self):
        processor = AcroTableCounts()
        assert processor.analysis_type == "acro_crosstab_count"
        assert "count" in processor.description.lower()
        assert processor.processing_query is None
        assert "2 columns" in processor.user_query_requirements

    def test_python_analysis(self):
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("not_recom", "convenient"),
            ("priority", "inconv"),
        ]
        mock_result.keys.return_value = ["recommendation", "finance"]

        mock_acro_instance = Mock()
        mock_acro_instance.finalise.return_value = None
        mock_acro_instance.crosstab.return_value.to_dict.return_value = {"col_a": {"row_1": 2}}

        with patch("five_safes_tes_analytics.node.local_processing.acro_module.ACRO", return_value=mock_acro_instance), \
             patch("five_safes_tes_analytics.node.local_processing.shutil.make_archive", return_value="/tmp/acro_output.zip"):
            processor = AcroTableCounts()
            result = processor.python_analysis(mock_result)

        assert isinstance(result, dict)
        assert "acro_output_zip" in result
        assert result["acro_status"] == "finalised"
        assert "table" in result
        mock_acro_instance.crosstab.assert_called_once()
        mock_acro_instance.finalise.assert_called_once()

    def test_crosstab_called_without_values(self):
        mock_result = Mock()
        mock_result.fetchall.return_value = [("A", "X"), ("B", "Y")]
        mock_result.keys.return_value = ["idx", "col"]

        mock_acro_instance = Mock()

        with patch("five_safes_tes_analytics.node.local_processing.acro_module.ACRO", return_value=mock_acro_instance), \
             patch("five_safes_tes_analytics.node.local_processing.shutil.make_archive", return_value="x.zip"):
            AcroTableCounts().python_analysis(mock_result)

        call_kwargs = mock_acro_instance.crosstab.call_args
        # count crosstab should not pass values= or aggfunc= keyword args
        assert "values" not in (call_kwargs.kwargs or {})
        assert "aggfunc" not in (call_kwargs.kwargs or {})
        assert call_kwargs.kwargs.get("margins") is True

    def test_registered_in_registry(self):
        registry = get_local_processing_registry()
        assert "acro_crosstab_count" in registry
        assert registry["acro_crosstab_count"] is AcroTableCounts


if __name__ == "__main__":
    pytest.main([__file__])
