"""Small local command-line interface for the evidence-based workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.workflows.agent_graph import run_agent_workflow


def main() -> None:
    """Run one question against optional local WDI and document inputs."""
    argument_parser = argparse.ArgumentParser(description="Ask the local World Bank evidence workflow.")
    argument_parser.add_argument("--question", required=True, help="Question to route to available agents")
    argument_parser.add_argument("--wdi-csv", default="data/processed/wdi_sample.csv")
    argument_parser.add_argument("--documents-folder", help="Folder containing local .txt/.md documents")
    argument_parser.add_argument(
        "--search-world-bank-documents", action="store_true",
        help="Search the public World Bank Documents & Reports API using the question",
    )
    argument_parser.add_argument("--json", action="store_true", help="Print the structured workflow response")
    arguments = argument_parser.parse_args()

    wdi_csv_path = Path(arguments.wdi_csv)
    workflow_response = run_agent_workflow(
        arguments.question,
        wdi_csv_path=wdi_csv_path if wdi_csv_path.is_file() else None,
        documents_folder=arguments.documents_folder,
        world_bank_document_query=arguments.question if arguments.search_world_bank_documents else None,
    )
    if arguments.json:
        print(json.dumps(workflow_response.model_dump(), indent=2, ensure_ascii=False, default=str))
        return

    print(f"Status: {workflow_response.status}")
    print(f"Agents: {', '.join(workflow_response.selected_agents)}")
    print(f"\n{workflow_response.answer}")
    if workflow_response.sources:
        print("\nSources:")
        for source_reference in workflow_response.sources:
            print(f"- {source_reference}")
    if workflow_response.limitations:
        print("\nLimitations:")
        for limitation in workflow_response.limitations:
            print(f"- {limitation}")


if __name__ == "__main__":
    main()
