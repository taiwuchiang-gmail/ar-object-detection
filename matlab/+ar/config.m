function cfg = config(objectName)
%CONFIG Resolve paths independently of the current working directory.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
configPath = fullfile(root,'configs','project.json');
if ~isfile(configPath)
    configPath = fullfile(root,'configs','project.example.json');
end
cfg = jsondecode(fileread(configPath));
cfg.root = root;
if nargin > 0
    assert(isfield(cfg.objects,objectName),'Unknown object: %s',objectName);
    cfg.object = cfg.objects.(objectName);
end
end
