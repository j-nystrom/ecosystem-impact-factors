import argparse
import importlib
import os
from box import Box

from core.utils.general_utils import create_logger  # create_run_folder_path

logger = create_logger(__name__)

# Load the config file into box object
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "dag_configs.yaml")
configs = Box.from_yaml(filename=config_path)


class BaseDAG:
    """Base DAG from which other DAGs inherit."""

    def __init__(self, name: str, tasks: list[type]) -> None:
        """Initialize the DAG with a name and list of tasks (classes)."""
        self.name = name
        self.tasks = tasks
        # self.run_folder_path = create_run_folder_path() <- Not implemented

    def run_dag(self) -> None:
        """Run the DAG by invoking and running tasks sequentially."""
        for task in self.tasks:
            task_instance = task()  # Not implemented: self.run_folder_path
            task_instance.run_task()


def create_dag(dag_config: Box) -> BaseDAG:
    """Create a DAG instance from configuration."""
    # Dynamically load tasks from their module paths
    name = dag_config.name
    tasks = [
        getattr(
            importlib.import_module(
                task.rsplit(".", 1)[0]
            ),  # Import module, e.g., core.data
            task.rsplit(".", 1)[1],  # Get class name, e.g., CurrentLandUseTask
        )
        for task in dag_config.tasks
    ]
    dag = BaseDAG(name, tasks)

    return dag


def main() -> None:
    """Parse command line arguments, create a DAG from config, and run it."""

    # Parse the DAG name from the command line
    parser = argparse.ArgumentParser(description="Run a specified DAG.")
    parser.add_argument("dag", help="Name of the DAG to run.")
    args = parser.parse_args()

    # Find the matching DAG config
    dag_config = next((dag for dag in configs.dags if dag.name == args.dag), None)
    if dag_config:
        dag = create_dag(dag_config)  # Create the DAG instance from config
        logger.info(f"Running {dag.name}.")
        try:
            dag.run_dag()  # Run the DAG
            logger.info(f"Successfully finished {dag.name}.")
        except Exception as e:
            logger.error(f"Error running DAG {dag.name}: {str(e)}")
    else:
        logger.warning(f"DAG named {args.dag} not found.")


if __name__ == "__main__":
    main()
