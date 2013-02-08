#GetEnv = True
#executable =  /afs/csail.mit.edu/system/common/matlab/2010a/bin/matlab
#executable =  /afs/csail.mit.edu/common/matlab/2010a/bin/matlab
#executable =  /data/zfs-scratch/matlab-2012a/bin/matlab
executable = /usr/bin/python
universe = vanilla
priority = 0
Notification = Never
# yodel machines have no scipy

requirements = OpSys == "LINUX"
#Requirements = Machine != "yodel7.csail.mit.edu" && Machine != "yodel4.csail.mit.edu" && Machine != "yodel2.csail.mit.edu" && Machine != "yodel10.csail.mit.edu" && Machine != "yodel7.csail.mit.edu" && Machine != "yodel6.csail.mit.edu" && Machine != "yodel8.csail.mit.edu" && Machine != "yodel17.csail.mit.edu" && Machine != "yodel13.csail.mit.edu" && Machine != "yodel12.csail.mit.edu" && Machine != "yodel11.csail.mit.edu"

#Requirements = isPublic && \
#               Memory >= 6144 && \
#               Cpus >= 1 && \

#Requirements = Arch == "X86_64"
Error = CondorOutput/err/$(PROCESS).err
Log = CondorOutput/log/$(PROCESS).log
Output = CondorOutput/out/$(PROCESS).out

queue 1
