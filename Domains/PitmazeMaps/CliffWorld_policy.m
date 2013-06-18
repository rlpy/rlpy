% Handcoded policy for the cliff world
% The policy is not optimal but it is safe
% Inputs:
% s: State
% agent, domain

function [a agent V prob] = CliffWorld_policy(s,agent,domain)

% Up    Down    Left    Right
% 1     2       3       4 

%agent.CL.internalModel.obj.noise
if agent.CL.internalModel.obj.noise > .05
    [a agent V prob] = CliffWorld_SafePolicy(s,agent,domain);
else
    [a agent V prob] = CliffWorld_AggressivePolicy(s,agent,domain);
end

end