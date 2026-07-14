"""langgraph nodes - orchestrator, he_agent, st_agent, alignment_agent, eval_agent.

phase 1 lands graph.py here: a StateGraph wiring orchestrator -> he ‖ st -> join
-> alignment -> eval, traced in langsmith. on the read path the agents resolve
cached refs (no compute); on the compute path they trigger pipeline stages.
"""