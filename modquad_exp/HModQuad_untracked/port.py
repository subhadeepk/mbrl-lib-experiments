function sysCall_init()
    simRemoteApi.start(19999)
    
end

function sysCall_actuation()
    -- put your actuation code here
end

function sysCall_sensing()
    -- put your sensing code here
end

function sysCall_cleanup()
    simRemoteApi.stop(19999)
    
    
end

-- See the user manual or the available code snippets for additional callback functions and details
