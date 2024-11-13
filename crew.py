from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, WebScrapingTool
import os
from dotenv import load_dotenv, find_dotenv

# Carrega as variáveis de ambiente
def load_env():
    _ = load_dotenv(find_dotenv())

# Função para obter a chave da API do modelo
def get_gemini_api_key():
    load_env()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    return gemini_api_key

# Configuração do modelo de linguagem
LLModel = LLM(
    model="gemini/gemini-1.5-pro",
    temperature=0.5,
    api_key=get_gemini_api_key(),
)

@CrewBase
class SistemaMultiAgentesCrew:
    """Sistema de Multi-Agentes aprimorado"""

    def __init__(self):
        # Configurações dos agentes
        self.agents_config = {
            'researcher': {
                "name": "Investigador da Web",
                "description": "Especialista em busca de dados e coleta de informações detalhadas.",
                "role": "Pesquisador",
                "goal": "Realizar buscas e coleta de dados da web para análise detalhada.",
                "backstory": "Veterano em investigações online com foco em coleta de informações e técnicas de scraping."
            },
            'fact_checker': {
                "name": "Verificador de Fatos",
                "description": "Especialista em validação e verificação de informações.",
                "role": "Verificador de Fatos",
                "goal": "Confirmar a precisão das informações coletadas pelo pesquisador.",
                "backstory": "Profissional com anos de experiência em checagem de fatos e análise crítica de dados."
            },
            'reporting_analyst': {
                "name": "Analista de Relatórios",
                "description": "Responsável por compilar os dados em relatórios compreensíveis.",
                "role": "Analista de Relatórios",
                "goal": "Compilar informações em relatórios finais claros e detalhados.",
                "backstory": "Analista de dados com expertise em transformar dados complexos em relatórios acessíveis."
            }
        }
        
        self.tasks_config = {
            'research_task': {
                "name": "Tarefa de Pesquisa",
                "description": "Realizar busca detalhada sobre o tópico especificado.",
                "expected_output": "Informações organizadas e relevantes sobre o tópico pesquisado."
            },
            'reporting_task': {
                "name": "Tarefa de Relatório",
                "description": "Organizar as informações pesquisadas em um relatório final.",
                "expected_output": "Relatório final completo com análise detalhada das informações coletadas."
            }
        }

    # Agente de pesquisa com ferramentas de busca e scraping
    @agent
    def researcher(self) -> Agent:
        return Agent(
            name=self.agents_config['researcher']["name"],
            description=self.agents_config['researcher']["description"],
            role=self.agents_config['researcher']["role"],
            goal=self.agents_config['researcher']["goal"],
            backstory=self.agents_config['researcher']["backstory"],
            tools=[SerperDevTool(), WebScrapingTool()],
            verbose=True,
            llm=LLModel
        )

    # Agente de checagem de fatos
    @agent
    def fact_checker(self) -> Agent:
        return Agent(
            name=self.agents_config['fact_checker']["name"],
            description=self.agents_config['fact_checker']["description"],
            role=self.agents_config['fact_checker']["role"],
            goal=self.agents_config['fact_checker']["goal"],
            backstory=self.agents_config['fact_checker']["backstory"],
            tools=[SerperDevTool()],  # Usa ferramenta de busca para auxiliar na verificação
            verbose=True,
            llm=LLModel
        )

    # Agente de relatório
    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            name=self.agents_config['reporting_analyst']["name"],
            description=self.agents_config['reporting_analyst']["description"],
            role=self.agents_config['reporting_analyst']["role"],
            goal=self.agents_config['reporting_analyst']["goal"],
            backstory=self.agents_config['reporting_analyst']["backstory"],
            verbose=True,
            llm=LLModel
        )

    # Tarefa de pesquisa
    @task
    def research_task(self) -> Task:
        return Task(
            name=self.tasks_config['research_task']["name"],
            description=self.tasks_config['research_task']["description"],
            expected_output=self.tasks_config['research_task']["expected_output"],
            verbose=True
        )

    # Tarefa de relatório
    @task
    def reporting_task(self) -> Task:
        return Task(
            name=self.tasks_config['reporting_task']["name"],
            description=self.tasks_config['reporting_task']["description"],
            expected_output=self.tasks_config['reporting_task']["expected_output"],
            output_file='results/report.md',
            verbose=True
        )

    # Configuração do Crew
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents={
                "researcher": self.researcher(),
                "fact_checker": self.fact_checker(),
                "reporting_analyst": self.reporting_analyst()
            },
            tasks={
                "research_task": self.research_task(),
                "reporting_task": self.reporting_task()
            },
            process=Process.sequential,
            verbose=True
        )

# Execução do sistema de multi-agentes
if __name__ == "__main__":
    sistema_crew = SistemaMultiAgentesCrew()
    crew_instance = sistema_crew.crew()
    crew_instance.run()
