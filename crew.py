from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os
from dotenv import load_dotenv, find_dotenv
from crewai_tools import ScrapeWebsiteTool


def load_env():
    _ = load_dotenv(find_dotenv())


def get_gemini_api_key():
    load_env()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    return gemini_api_key


LLModel = LLM(
    model="gemini/gemini-1.5-pro",
    temperature=0.5,
    api_key=get_gemini_api_key(),
)

# Uncomment the following line to use an example of a custom tool
# from sistema_multi_agentes.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool


@CrewBase
class SistemaMultiAgentesCrew:
    """SistemaMultiAgentes crew"""

    @agent
    def pesquisador_senior(self) -> Agent:
        return Agent(
            config=self.agents_config["pesquisador_senior"],
            tool=[ScrapeWebsiteTool()],
            verbose=True,
            llm=LLModel,
        )

    @agent
    def checador_de_fatos(self) -> Agent:
        return Agent(
            config=self.agents_config["checador_de_fatos"],
            tools=[ScrapeWebsiteTool()],
            verbose=True,
            llm=LLModel,
        )

    @agent
    def analista_de_relatorios(self) -> Agent:
        return Agent(
            config=self.agents_config["analista_de_relatorios"],
            verbose=True,
            llm=LLModel,
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"], verbose=True)

    @task
    def fact_checking_task(self) -> Task:
        return Task(config=self.tasks_config["fact_checking_task"], verbose=True)

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_task"],
            output_file="results/report1.html",
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the SistemaMultiAgentes crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )