
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os
from dotenv import load_dotenv, find_dotenv


def load_env():
    _ = load_dotenv(find_dotenv())


def get_gemini_api_key():
    load_env()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    return gemini_api_key


LLModel = LLM(
    model="gemini/gemini-1.5-pro",  # Modelo de linguagem configurado
    temperature=0.5,                 # Controla a aleatoriedade das respostas
    api_key=get_gemini_api_key(),    # Chave de API para autenticação
)

# Ferramentas fictícias de busca e scraping
from crewai_tools import SerperDevTool, WebScraperTool  # Exemplo de ferramentas

# Classe principal do sistema de multiagentes
@CrewBase
class SistemaMultiAgentesCrew():

    # Configuração de variáveis de backstory para os agentes
    agents_config = {
        'researcher': {
            'description': 'Pesquisador Cético que examina rigorosamente a confiabilidade das fontes.',
            'objective': 'Realizar buscas detalhadas e verificar a credibilidade das fontes.',
        },
        'reporting_analyst': {
            'description': 'Analista Estratégico, prioriza dados para recomendações de alto impacto.',
            'objective': 'Gerar relatórios orientados para decisões estratégicas.',
        },
        'fact_checker': {
            'description': 'Verificador de Fatos Imparcial, dedicado a checar a veracidade de cada informação.',
            'objective': 'Identificar e confirmar a autenticidade das informações coletadas.',
        }
    }

    # Definição do agente pesquisador com ferramentas de busca e scraping
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],    # Usa a configuração do pesquisador
            tools=[SerperDevTool(), WebScraperTool()],  # Ferramentas adicionadas para busca e scraping
            verbose=True,                               # Imprime logs detalhados para debug
            llm=LLModel                                 # Modelo de linguagem para auxiliar o agente
        )

    # Definição do agente analista de relatórios
    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],  # Configuração do analista
            verbose=True,                                     # Logs detalhados
            llm=LLModel                                       # Modelo de linguagem
        )

    # Definição do agente de checagem de fatos para verificação de informações
    @agent
    def fact_checker(self) -> Agent:
        return Agent(
            config=self.agents_config['fact_checker'],  # Configuração do checador de fatos
            tools=[SerperDevTool()],                    # Ferramenta para checagem de fatos (exemplo)
            verbose=True,                               # Logs detalhados
            llm=LLModel                                 # Modelo de linguagem
        )

    # Definição da tarefa de pesquisa para o agente pesquisador
    @task
    def research_task(self) -> Task:
        return Task(
            config={'name': 'Pesquisa de informações relevantes'},
            verbose=True  # Logs detalhados
        )

    # Definição da tarefa de relatório, onde os dados são processados para um output final
    @task
    def reporting_task(self) -> Task:
        return Task(
            config={'name': 'Geração de relatório final com insights'},
            output_file='results/report.md',  # Define o arquivo de saída do relatório
            verbose=True                      # Logs detalhados
        )

    # Criação do grupo de agentes (crew) e tarefas para definir o fluxo do sistema
    @crew
    def crew(self) -> Crew:
        """Cria o crew do SistemaMultiAgentes"""
        return Crew(
            agents=self.agents,                # Agentes definidos com @agent decorator
            tasks=self.tasks,                  # Tarefas definidas com @task decorator
            process=Process.sequential,        # Define o processo como sequencial
            verbose=True                       # Logs detalhados do processo
        )
