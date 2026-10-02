# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %pip install langgraph

# COMMAND ----------

# Databricks notebook source
# 05_supervisor_workflow.py
# Multi-agent supervisor workflow for the World Bank Intelligence project.
#
# Prerequisites:
# 1. Run 02_delta_medallion.py successfully.
# 2. Confirm the Gold table exists at workspace.default.indicators_gold.
# 3. Install/enable langgraph in the notebook environment if it is not already available.

from typing import Any, TypedDict, Optional
import os
import re
import time
import uuid
import json
from datetime import datetime, timezone

import requests

from pyspark.sql import functions as F
from langgraph.graph import StateGraph, END
from src.utils.config import get_table_name
from src.decision import DecisionInput, PrototypeDecisionPolicy



# 1. Project configuration


CATALOG = os.getenv("DATABRICKS_CATALOG", "workspace")
SCHEMA = os.getenv("DATABRICKS_SCHEMA", "default")
os.environ["DATABRICKS_CATALOG"] = CATALOG
os.environ["DATABRICKS_SCHEMA"] = SCHEMA
GOLD_TABLE = get_table_name("indicators_gold")
WORKFLOW_LOG_TABLE = get_table_name("workflow_logs")

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"Gold table: {GOLD_TABLE}")



# 2. Workflow state


class IntelligenceWorkflowState(TypedDict, total=False):
    question: str
    country_code: str
    indicator_code: str
    selected_agent: str
    status: str
    answer: str
    evidence: list[dict[str, Any]]
    validation_status: str
    validation_message: str
    execution_id: str
    started_at_utc: str
    latency_seconds: float
    error: str



# 3. Query understanding and routing


COUNTRY_ALIASES = {
    "united kingdom": "GBR",
    "uk": "GBR",
    "britain": "GBR",
    "great britain": "GBR",
    "england": "GBR",
    "india": "IND",
    "united states": "USA",
    "us": "USA",
    "usa": "USA",
    "united states of america": "USA",
    "canada": "CAN",
    "germany": "DEU",
    "france": "FRA",
    "japan": "JPN",
    "china": "CHN",
    "australia": "AUS",
}

INDICATOR_ALIASES = {
    "gdp": ("NY.GDP.MKTP.CD", "GDP (current US$)"),
    "gross domestic product": ("NY.GDP.MKTP.CD", "GDP (current US$)"),
    "population": ("SP.POP.TOTL", "Population, total"),
    "unemployment": ("SL.UEM.TOTL.ZS", "Unemployment, total (% of total labor force)"),
    "unemployment rate": ("SL.UEM.TOTL.ZS", "Unemployment, total (% of total labor force)"),
}

def extract_country_code(question: str) -> Optional[str]:
    q = question.lower()
    # Match longer aliases first, so "United States" is checked before "US".
    for alias in sorted(COUNTRY_ALIASES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            return COUNTRY_ALIASES[alias]
    # Accept a literal three-character ISO-like country code when explicitly typed.
    match = re.search(r"\b([A-Z]{3})\b", question)
    return match.group(1) if match else None

def extract_indicator(question: str) -> tuple[Optional[str], Optional[str]]:
    q = question.lower()
    for alias in sorted(INDICATOR_ALIASES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            code, name = INDICATOR_ALIASES[alias]
            return code, name

    # Also accept a WDI indicator code included directly in the question.
    match = re.search(r"\b[A-Z]{2}\.[A-Z0-9]{3,}(?:\.[A-Z0-9]{2,})+\b", question.upper())
    if match:
        return match.group(0), match.group(0)
    return None, None

def route_question(question: str) -> str:
    q = question.lower()

    economic_terms = (
        "gdp", "gross domestic product", "population", "unemployment",
        "economic", "economy", "growth", "indicator", "inflation",
        "income", "development data",
    )
    finance_terms = ("finance", "financial", "loan", "funding", "disbursement")
    project_terms = ("project", "investment operation", "world bank operation")
    research_terms = ("research", "report", "document", "publication", "evidence")

    # Check specialist intents before generic economic words such as "data"
    # or "indicator", so financial/project/research questions reach their agents.
    if any(term in q for term in finance_terms):
        return "financial_data_agent"
    if any(term in q for term in project_terms):
        return "project_data_agent"
    if any(term in q for term in research_terms):
        return "research_data_agent"
    if any(term in q for term in economic_terms):
        return "economic_data_agent"
    return "clarification_agent"



# 4. Economic Data Agent: retrieves evidence from the Gold Delta table


def economic_data_agent(
    country_code: str,
    indicator_code: str,
    limit: int = 10,
) -> dict[str, Any]:
    country = country_code.strip().upper()
    indicator = indicator_code.strip().upper()

    if not re.fullmatch(r"[A-Z0-9]{3}", country):
        return {
            "status": "invalid_input",
            "answer": "The country code must be a three-character code, such as GBR or IND.",
            "evidence": [],
        }

    if not re.fullmatch(r"[A-Z]{2}\.[A-Z0-9]{3,}(?:\.[A-Z0-9]{2,})+", indicator):
        return {
            "status": "invalid_input",
            "answer": "The indicator code does not look like a valid World Bank indicator code.",
            "evidence": [],
        }

    try:
        records = (
            spark.table(GOLD_TABLE)
            .filter(F.upper(F.col("country_code")) == country)
            .filter(F.upper(F.col("indicator_code")) == indicator)
            .filter(F.col("value").isNotNull())
            .orderBy(F.col("year").desc())
            .limit(max(1, min(int(limit), 100)))
            .collect()
        )
    except Exception as exc:
        return {
            "status": "data_error",
            "answer": (
                f"Could not read the Gold table '{GOLD_TABLE}'. "
                "Confirm that 02_delta_medallion completed successfully and that "
                "the table name is correct."
            ),
            "evidence": [],
            "error": str(exc),
        }

    if not records:
        return {
            "status": "no_data",
            "answer": (
                f"No non-null observations were found for country {country} "
                f"and indicator {indicator} in the Gold table."
            ),
            "evidence": [],
        }

    evidence = []
    for row in records:
        evidence.append({
            "country_code": row["country_code"],
            "country": row["country"],
            "indicator_code": row["indicator_code"],
            "indicator_name": row["indicator_name"],
            "year": int(row["year"]) if row["year"] is not None else None,
            "value": float(row["value"]) if row["value"] is not None else None,
            "source_url": row["source_url"],
        })

    latest = evidence[0]
    indicator_name = latest.get("indicator_name") or indicator
    country_name = latest.get("country") or country
    value_text = f"{latest['value']:,.4f}".rstrip("0").rstrip(".")
    answer = (
        f"The latest available {indicator_name} observation for {country_name} "
        f"is {value_text} for {latest['year']}. "
        f"The result is retrieved from the project's World Bank Gold table."
    )

    return {
        "status": "success",
        "answer": answer,
        "country_code": country,
        "indicator_code": indicator,
        "evidence": evidence,
    }



# 5. Specialist agents and workflow nodes


FINANCIAL_INDICATORS = {
    "domestic credit": ("FS.AST.PRVT.GD.ZS", "Domestic credit to private sector (% of GDP)"),
    "private sector credit": ("FS.AST.PRVT.GD.ZS", "Domestic credit to private sector (% of GDP)"),
    "lending interest": ("FR.INR.LEND", "Lending interest rate (%)"),
    "lending rate": ("FR.INR.LEND", "Lending interest rate (%)"),
    "foreign direct investment": ("BX.KLT.DINV.WD.GD.ZS", "Foreign direct investment, net inflows (% of GDP)"),
    "fdi": ("BX.KLT.DINV.WD.GD.ZS", "Foreign direct investment, net inflows (% of GDP)"),
    "inflation": ("FP.CPI.TOTL.ZG", "Inflation, consumer prices (annual %)"),
    "consumer price inflation": ("FP.CPI.TOTL.ZG", "Inflation, consumer prices (annual %)"),
}

def _records_from_pandas(frame) -> list[dict[str, Any]]:
    """Convert the World Bank ingestion helper's pandas DataFrame to JSON-safe rows."""
    if frame is None or frame.empty:
        return []
    rows = frame.to_dict(orient="records")
    output = []
    for row in rows:
        output.append({
            "country_code": str(row.get("country_code", "")),
            "country": str(row.get("country", "")),
            "indicator_code": str(row.get("indicator_code", "")),
            "indicator_name": str(row.get("indicator_name", "")),
            "year": int(row["year"]) if row.get("year") is not None else None,
            "value": float(row["value"]) if row.get("value") is not None else None,
            "source_url": row.get("source_url"),
        })
    return output

def financial_data_agent(question: str, country_code: str, limit: int = 10) -> dict[str, Any]:
    """Retrieve financial/economic finance indicators from the World Bank WDI API."""
    q = question.lower()
    selected = None
    for alias in sorted(FINANCIAL_INDICATORS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            selected = FINANCIAL_INDICATORS[alias]
            break

    # If the question does not name a specific finance metric, use the configured
    # domestic-credit indicator as a transparent default rather than inventing a result.
    if selected is None:
        selected = FINANCIAL_INDICATORS["domestic credit"]

    indicator_code, indicator_name = selected
    try:
        frame = fetch_indicators(
            country_code,
            [indicator_code],
            base_url=os.getenv("WB_API_BASE", "https://api.worldbank.org/v2"),
            timeout=30,
            max_retries=3,
        )
        evidence = _records_from_pandas(frame)
    except Exception as exc:
        return {
            "status": "data_error",
            "answer": f"Unable to retrieve World Bank financial indicator data: {exc}",
            "evidence": [],
            "error": str(exc),
            "selected_agent": "financial_data_agent",
        }

    evidence = [item for item in evidence if item.get("value") is not None]
    evidence.sort(key=lambda item: item.get("year") or 0, reverse=True)
    evidence = evidence[:max(1, min(int(limit), 100))]

    if not evidence:
        return {
            "status": "no_data",
            "answer": (
                f"No observations were returned for {indicator_name} "
                f"({indicator_code}) and country {country_code}."
            ),
            "evidence": [],
            "selected_agent": "financial_data_agent",
        }

    latest = evidence[0]
    value_text = f"{latest['value']:,.4f}".rstrip("0").rstrip(".")
    return {
        "status": "success",
        "answer": (
            f"The latest available {indicator_name} observation for "
            f"{latest.get('country') or country_code} is {value_text} "
            f"for {latest['year']}. The data was retrieved from the World Bank "
            "World Development Indicators API."
        ),
        "country_code": country_code,
        "indicator_code": indicator_code,
        "evidence": evidence,
        "selected_agent": "financial_data_agent",
    }

def project_data_agent(question: str, country_code: Optional[str] = None, limit: int = 10) -> dict[str, Any]:
    """Search World Bank project records through the public Projects API."""
    params = {"format": "json", "per_page": max(1, min(int(limit) * 5, 100))}
    if country_code:
        params["countrycode"] = country_code

    try:
        response = requests.get(
            "https://api.worldbank.org/v2/projects",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        project_rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        if not isinstance(project_rows, list):
            project_rows = []
    except Exception as exc:
        return {
            "status": "data_error",
            "answer": f"Unable to retrieve World Bank project records: {exc}",
            "evidence": [],
            "error": str(exc),
            "selected_agent": "project_data_agent",
        }

    # Filter the returned project list locally using meaningful terms from the question.
    stop_words = {"what", "which", "show", "list", "latest", "world", "bank", "project", "projects", "for", "in", "the", "and", "about"}
    query_terms = [term for term in re.findall(r"[a-z0-9]+", question.lower()) if len(term) > 2 and term not in stop_words]
    matches = []
    for item in project_rows:
        if not isinstance(item, dict):
            continue
        name = item.get("project_name") or item.get("projectname") or item.get("name") or ""
        country = item.get("countryname") or item.get("country_name") or ""
        searchable = f"{name} {country} {item.get('id', '')}".lower()
        if not query_terms or any(term in searchable for term in query_terms):
            matches.append({
                "project_id": item.get("id") or item.get("project_id"),
                "project_name": name or "Project name not supplied",
                "country": country,
                "country_code": country_code,
                "status": item.get("status"),
                "approval_date": item.get("boardapprovaldate") or item.get("board_approval_date"),
                "total_amount": item.get("totalamt") or item.get("total_amount"),
                "url": item.get("url") or item.get("project_url"),
                "source_url": response.url,
            })
        if len(matches) >= limit:
            break

    if not matches:
        return {
            "status": "no_data",
            "answer": (
                "No matching projects were found in the World Bank Projects API "
                "response. Try a country name or a broader project keyword."
            ),
            "evidence": [],
            "selected_agent": "project_data_agent",
        }

    lines = [
        f"{i}. {item['project_name']} (ID: {item.get('project_id') or 'not supplied'}; "
        f"Country: {item.get('country') or 'not supplied'}; "
        f"Status: {item.get('status') or 'not supplied'})"
        for i, item in enumerate(matches, start=1)
    ]
    return {
        "status": "success",
        "answer": "World Bank project records matching the request:\n" + "\n".join(lines),
        "evidence": matches,
        "selected_agent": "project_data_agent",
    }

def research_data_agent(question: str, limit: int = 5) -> dict[str, Any]:
    """Search World Bank Documents & Reports (WDS) for related publications."""
    params = {
        "format": "json",
        "qterm": question[:300],
        "rows": max(1, min(int(limit), 20)),
        "os": 0,
        "srt": "rel",
    }
    try:
        response = requests.get(
            "https://search.worldbank.org/api/v2/wds",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {
            "status": "data_error",
            "answer": f"Unable to search World Bank Documents & Reports: {exc}",
            "evidence": [],
            "error": str(exc),
            "selected_agent": "research_data_agent",
        }

    docs = payload.get("documents", {}) if isinstance(payload, dict) else {}
    if isinstance(docs, dict):
        doc_rows = list(docs.values())
    elif isinstance(docs, list):
        doc_rows = docs
    else:
        doc_rows = []

    evidence = []
    for doc in doc_rows:
        if not isinstance(doc, dict):
            continue
        title = doc.get("display_title") or doc.get("title") or doc.get("docdt") or "World Bank publication"
        url = doc.get("url") or doc.get("pdfurl") or doc.get("docurl")
        evidence.append({
            "document_id": doc.get("id") or doc.get("docid"),
            "title": title,
            "date": doc.get("display_date") or doc.get("docdt") or doc.get("date"),
            "abstract": doc.get("abstract") or doc.get("sub_title") or doc.get("description"),
            "country": doc.get("countryname") or doc.get("country"),
            "document_type": doc.get("doctype") or doc.get("doc_type"),
            "url": url,
            "source_url": response.url,
        })
        if len(evidence) >= limit:
            break

    if not evidence:
        return {
            "status": "no_data",
            "answer": (
                "The World Bank Documents & Reports search returned no usable "
                "document records for this query. Try a shorter or more specific search phrase."
            ),
            "evidence": [],
            "selected_agent": "research_data_agent",
        }

    answer_lines = []
    for i, doc in enumerate(evidence, start=1):
        line = f"{i}. {doc['title']}"
        if doc.get("date"):
            line += f" ({doc['date']})"
        if doc.get("url"):
            line += f" — {doc['url']}"
        answer_lines.append(line)

    return {
        "status": "success",
        "answer": "Relevant World Bank publications found:\n" + "\n".join(answer_lines),
        "evidence": evidence,
        "selected_agent": "research_data_agent",
    }


def supervisor_node(state: IntelligenceWorkflowState) -> dict[str, Any]:
    question = (state.get("question") or "").strip()
    if not question:
        return {
            "selected_agent": "clarification_agent",
            "status": "needs_clarification",
            "answer": "Please enter a question about an economic indicator, country, project, or World Bank publication.",
            "evidence": [],
        }
    return {"selected_agent": route_question(question), "status": "routed"}

def economic_node(state: IntelligenceWorkflowState) -> dict[str, Any]:
    question = state.get("question", "")
    country = state.get("country_code") or extract_country_code(question)
    indicator = state.get("indicator_code")
    indicator_name = None
    if not indicator:
        indicator, indicator_name = extract_indicator(question)

    if not country or not indicator:
        missing = []
        if not country:
            missing.append("country (for example, United Kingdom or India)")
        if not indicator:
            missing.append("indicator (for example, GDP, population, or unemployment)")
        return {
            "status": "needs_clarification",
            "answer": "Please specify the " + " and ".join(missing) + ".",
            "evidence": [],
            "selected_agent": "economic_data_agent",
        }

    result = economic_data_agent(country, indicator)
    result["selected_agent"] = "economic_data_agent"
    if indicator_name:
        result["requested_indicator_name"] = indicator_name
    return result

def financial_node(state: IntelligenceWorkflowState) -> dict[str, Any]:
    country = state.get("country_code") or extract_country_code(state.get("question", ""))
    if not country:
        return {
            "status": "needs_clarification",
            "answer": "Please specify a country, such as the United Kingdom or India, for the financial indicator query.",
            "evidence": [],
            "selected_agent": "financial_data_agent",
        }
    return financial_data_agent(state.get("question", ""), country)

def project_node(state: IntelligenceWorkflowState) -> dict[str, Any]:
    country = state.get("country_code") or extract_country_code(state.get("question", ""))
    return project_data_agent(state.get("question", ""), country)

def research_node(state: IntelligenceWorkflowState) -> dict[str, Any]:
    return research_data_agent(state.get("question", ""))

def clarification_node(state: IntelligenceWorkflowState) -> dict[str, Any]:
    return {
        "selected_agent": "clarification_agent",
        "status": "needs_clarification",
        "answer": (
            "I could not identify a supported data task. Ask about a country and "
            "an economic indicator, a World Bank project, or a World Bank report. "
            "Example: 'List World Bank projects in the United Kingdom.'"
        ),
        "evidence": [],
    }



# 6. Evidence validation


def validate_evidence_node(state: IntelligenceWorkflowState) -> dict[str, Any]:
    status = state.get("status", "unknown")
    evidence = state.get("evidence") or []

    if status != "success":
        return {
            "validation_status": "not_applicable",
            "validation_message": f"Evidence validation skipped because status is '{status}'.",
        }

    agent = state.get("selected_agent", "")
    if agent == "economic_data_agent" or agent == "financial_data_agent":
        required_fields = {"country_code", "indicator_code", "year", "value", "source_url"}
    elif agent == "project_data_agent":
        required_fields = {"project_name", "source_url"}
    elif agent == "research_data_agent":
        required_fields = {"title", "source_url"}
    else:
        required_fields = set()

    if not evidence:
        return {
            "status": "validation_failed",
            "validation_status": "failed",
            "validation_message": "The agent returned success but supplied no evidence.",
            "answer": "The result could not be validated because no supporting evidence was returned.",
        }

    invalid = []
    for index, item in enumerate(evidence):
        missing = required_fields - set(item.keys())
        if missing or item.get("value") is None or item.get("year") is None:
            invalid.append({"record": index, "missing_fields": sorted(missing)})

    if invalid:
        return {
            "status": "validation_failed",
            "validation_status": "failed",
            "validation_message": f"Evidence records are incomplete: {invalid}",
            "answer": "The result could not be validated because one or more evidence records are incomplete.",
        }

    return {
        "validation_status": "passed",
        "validation_message": f"Validated {len(evidence)} evidence record(s) for the {agent} response.",
    }



# 7. Build the LangGraph workflow


def select_agent_route(state: IntelligenceWorkflowState) -> str:
    return state.get("selected_agent", "clarification_agent")

workflow_builder = StateGraph(IntelligenceWorkflowState)

workflow_builder.add_node("supervisor", supervisor_node)
workflow_builder.add_node("economic_data_agent", economic_node)
workflow_builder.add_node("financial_data_agent", financial_node)
workflow_builder.add_node("project_data_agent", project_node)
workflow_builder.add_node("research_data_agent", research_node)
workflow_builder.add_node("clarification_agent", clarification_node)
workflow_builder.add_node("validate_evidence", validate_evidence_node)

workflow_builder.set_entry_point("supervisor")

workflow_builder.add_conditional_edges(
    "supervisor",
    select_agent_route,
    {
        "economic_data_agent": "economic_data_agent",
        "financial_data_agent": "financial_data_agent",
        "project_data_agent": "project_data_agent",
        "research_data_agent": "research_data_agent",
        "clarification_agent": "clarification_agent",
    },
)

# All routes proceed to evidence validation; non-success statuses are marked
# not_applicable by the validator.
for node_name in (
    "economic_data_agent",
    "financial_data_agent",
    "project_data_agent",
    "research_data_agent",
    "clarification_agent",
):
    workflow_builder.add_edge(node_name, "validate_evidence")

workflow_builder.add_edge("validate_evidence", END)

intelligence_workflow = workflow_builder.compile()

print("LangGraph supervisor workflow compiled successfully.")



# 8. Public workflow runner with timing and execution ID


def run_supervisor(question: str) -> dict[str, Any]:
    execution_id = str(uuid.uuid4())
    started = time.perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat()

    initial_state: IntelligenceWorkflowState = {
        "question": question,
        "execution_id": execution_id,
        "started_at_utc": started_at_utc,
    }

    try:
        result = intelligence_workflow.invoke(initial_state)
        evidence = result.get("evidence") or []
        policy_decision = PrototypeDecisionPolicy().decide(DecisionInput(
            execution_id=execution_id, question=question,
            candidate_agents=[result.get("selected_agent", "")],
            allowed_agents=["economic_data_agent", "financial_data_agent", "project_data_agent", "research_data_agent"],
            available_evidence=evidence, evidence_coverage=1.0 if evidence else 0.0,
            task_status=result.get("status", "unknown"), retrieval_attempts=2,
            max_retrieval_attempts=2, remaining_steps=1,
        ))
        result["decision"] = policy_decision.model_dump()
        result["execution_id"] = execution_id
        result["started_at_utc"] = started_at_utc
        result["latency_seconds"] = round(time.perf_counter() - started, 4)
        return result
    except Exception as exc:
        return {
            "question": question,
            "execution_id": execution_id,
            "started_at_utc": started_at_utc,
            "selected_agent": "workflow_error",
            "status": "error",
            "answer": "The supervisor workflow failed while processing the request.",
            "evidence": [],
            "validation_status": "not_completed",
            "error": str(exc),
            "latency_seconds": round(time.perf_counter() - started, 4),
        }



# 9. Optional workflow logging to a Delta table


def log_workflow_result(result: dict[str, Any]) -> str:
    """Append one workflow result to a Delta table; returns the table name."""
    log_schema = """
        execution_id STRING,
        timestamp_utc STRING,
        question STRING,
        selected_agent STRING,
        status STRING,
        validation_status STRING,
        latency_seconds DOUBLE,
        evidence_count INT,
        answer STRING,
        evidence_json STRING,
        error STRING
    """

    spark.sql(f"CREATE TABLE IF NOT EXISTS {WORKFLOW_LOG_TABLE} ({log_schema}) USING DELTA")

    evidence = result.get("evidence") or []
    row = [(
        result.get("execution_id"),
        datetime.now(timezone.utc).isoformat(),
        result.get("question"),
        result.get("selected_agent"),
        result.get("status"),
        result.get("validation_status"),
        float(result.get("latency_seconds") or 0.0),
        int(len(evidence)),
        result.get("answer"),
        json.dumps(evidence, default=str),
        result.get("error"),
    )]

    columns = [
        "execution_id", "timestamp_utc", "question", "selected_agent",
        "status", "validation_status", "latency_seconds", "evidence_count",
        "answer", "evidence_json", "error",
    ]
    log_df = spark.createDataFrame(row, schema=columns)
    log_df.write.format("delta").mode("append").saveAsTable(WORKFLOW_LOG_TABLE)
    return WORKFLOW_LOG_TABLE



# 10. Run a test


test_question = "What is the latest available GDP data for the United Kingdom?"
test_result = run_supervisor(test_question)

print("Execution ID:", test_result.get("execution_id"))
print("Selected agent:", test_result.get("selected_agent"))
print("Status:", test_result.get("status"))
print("Validation:", test_result.get("validation_status"))
print("Latency (seconds):", test_result.get("latency_seconds"))
print("Answer:", test_result.get("answer"))

if test_result.get("evidence"):
    print("Evidence records:", len(test_result["evidence"]))
    display(spark.createDataFrame(test_result["evidence"]))
else:
    print("No evidence returned.")

# Optional: enable logging after the workflow test succeeds.
# Uncomment the next two lines to write the test result to Delta.
# log_table = log_workflow_result(test_result)
# print(f"Workflow result logged to: {log_table}")
