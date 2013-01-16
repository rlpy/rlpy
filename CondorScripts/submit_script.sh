#GetEnv = True
#executable =  /afs/csail.mit.edu/system/common/matlab/2010a/bin/matlab
#executable =  /afs/csail.mit.edu/common/matlab/2010a/bin/matlab
#executable =  /data/zfs-scratch/matlab-2012a/bin/matlab 
executable = /usr/bin/python
universe = vanilla
priority = 0
Notification = Never

#Requirements = isPublic && \
#               Memory >= 6144 && \ 
#               Cpus >= 1 && \

#Requirements = Arch == "X86_64"
#Requirements = OpSys =="LINUX"
#Requirements = isTIDOR==true
#Requirements = DebianVersion==5.0

#executable = /afs/csail.mit.edu/system/amd64_linux26/matlab/latest/bin/matlab
#should_transfer_files = IF_NEEDED
#WhenToTransferOutput = ON_EXIT

#Error = CondorOutput/err/$(PROCESS).err
#Log = CondorOutput/log/$(PROCESS).log
#Output = CondorOutput/out/$(PROCESS).out

queue 1
