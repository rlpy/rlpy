% Handcoded policy for the cliff world
% The policy is not optimal but it is safe
% Inputs:
% s: State
% agent, domain

function [a agent V prob] = CliffWorldLarge_SafePolicy(s,agent,domain)

% Up    Down    Left    Right
% 1     2       3       4 


policy=[
4 4 4 4 4 4 4 4 4 2
1 1 1 1 1 1 1 1 1 2
1 3 1 1 1 1 1 1 4 2
1 3 3 1 1 1 1 4 4 2
1 3 3 3 1 1 4 4 4 2
1 3 3 3 3 4 4 4 4 2
1 3 3 3 3 4 4 4 4 2
1 3 3 3 3 4 4 4 4 2
1 3 1 1 1 1 1 1 4 2
1 1 1 1 1 1 1 1 1 2
];  



a = policy(s(1),s(2));  
V = 0; %Dont know the value but pass 0
prob = 1;

end