from crewai import Agent, Task, Crew, Process, LLM
from crewai.tasks.task_output import TaskOutput
import os, json
import datetime

llm = LLM(
    model="ollama/gemma3n:latest",
    base_url="http://localhost:11434"
)

# Define a multimodal agent with vision capabilities
researcher = Agent(
    role="Product Quality Inspector",
    goal="Analyze product {image} and report on quality attributes",
    backstory="Senior visual inspector with extensive industry knowledge in product quality inspection",
    llm=llm,
    multimodal=True,
    verbose=True
)

# Create a task that involves image analysis
task = Task(
    description="Analyze the product image at {image} and provide a detailed report",
    expected_output="A detailed report on quality assessment with today's {date}.",
    agent=researcher
)

# Assemble the crew and execute tasks sequentially
crew = Crew(
    agents=[researcher],
    tasks=[task],
    process=Process.sequential,
    verbose=True
)

inputs = {
    'image': 'https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhL0ZIm6wh06croVC6i2mo_vWm6MDYZPgd5h0jXQGB13-4Z1ugRQD5OuF9HnZEOfe6mC_62S_bPiOIocO1ljytrOfwxNsOynJO8TYzJw31NkfG4cVwOJ-kKnrTBtZ_wC2A3YuAdaVOO7YA/s1600/broken+cup+2.jpg',
    'date': datetime.datetime.now().strftime("%Y-%m-%d")
}

result = crew.kickoff(inputs=inputs)
print(result)