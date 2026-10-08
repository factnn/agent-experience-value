"""Multi-user-turn BFCL controller, independent of generation/training backend.

Reveals only the current user turn. Reference actions and grades remain server
side; tool responses alone return to the conversation. Not yet a TRL adapter.
"""
import copy
from bfcl_safe_runtime import SimulatorEpisode


class BFCLConversation:
    def __init__(self,task,reference_turns):
        if len(task['question'])!=len(reference_turns):raise ValueError('question/answer turn count mismatch')
        self.task=copy.deepcopy(task);self.reference_turns=copy.deepcopy(reference_turns)
        self.actual=SimulatorEpisode(task);self.reference=SimulatorEpisode(task)
        self.turn=0;self.history=copy.deepcopy(task['question'][0]);self.events=[];self.grades=[]

    @property
    def completed(self):return self.turn==len(self.task['question'])

    def messages(self):return copy.deepcopy(self.history)

    def assistant_message(self,message):
        if self.completed:raise RuntimeError('episode already ended')
        if message.get('role')!='assistant':raise ValueError('expected assistant message')
        self.history.append(copy.deepcopy(message))

    def tool_call(self,call,name=None):
        if self.completed:raise RuntimeError('episode already ended')
        try:response=self.actual.call(call)
        except Exception as error:
            # Simulator methods may raise ordinary execution errors, e.g.
            # missing files. They are feedback, never a crashed policy episode.
            response=f'Error during execution: {error}'
            self.actual.responses.append(response)
        self.events.append({'turn':self.turn,'call':call,'response':response})
        self.history.append({'role':'tool','name':name or call.split('(',1)[0],'content':response})
        return response

    def finish_user_turn(self,final_message):
        self.assistant_message(final_message)
        reference_calls=self.reference_turns[self.turn]
        self.reference.turn_response_count=len(reference_calls)
        for call in reference_calls:self.reference.call(call)
        result=self.actual.matches(self.reference,self.turn)
        has_calls=any(e['turn']==self.turn for e in self.events)
        valid=bool(result['valid']) and (has_calls or not reference_calls)
        # Latch failures at the correct user boundary. Later repairs cannot erase
        # a previous turn's failed official state/response check.
        self.grades.append({'turn':self.turn,'valid':valid,'check':result})
        self.turn+=1
        if self.completed:return []
        next_messages=copy.deepcopy(self.task['question'][self.turn])
        self.history.extend(next_messages)
        return next_messages

    def reward(self):return float(self.completed and all(g['valid'] for g in self.grades))
