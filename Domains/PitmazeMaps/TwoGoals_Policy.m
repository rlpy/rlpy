% The policy is not optimal but it is safe
% Inputs:
% s: State
% agent, domain

function [a agent V prob] = TwoGoals_Policy(s,agent,domain)

% Up    Down    Left    Right
% 1     2       3       4 


%agent.CL.internalModel.obj.noise
if agent.CL.internalModel.obj.noise > agent.CL.AcceptableNoise 
    [a agent V prob] = TwoGoals_SafePolicy(s,agent,domain);
    if agent.CL.lastPolicyWasAggressive
        agent.CL.lastPolicyWasAggressive = 0;
        agent.CL.visits(:) = 0; %Reset counts
    end    
else
    [a agent V prob] = TwoGoals_AggressivePolicy(s,agent,domain);
    if ~agent.CL.lastPolicyWasAggressive
        agent.CL.lastPolicyWasAggressive = 1;
        agent.CL.visits(:) = 0; %Reset counts
    end    
end

end