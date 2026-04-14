FROM python:3.12

# Set working directory inside container
WORKDIR /app

# Copy the entire package
COPY src/five_safes_tes_analytics /app/five_safes_tes_analytics

# Install dependencies
RUN pip install click sqlalchemy psycopg2-binary numpy tdigest pandas acro

# Entry point
ENTRYPOINT ["python", "-m", "five_safes_tes_analytics.node.query_resolver"]
