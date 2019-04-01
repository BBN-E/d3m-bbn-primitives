class PipelinePartitionSubstitution(object):
    def __init__(self, orig_primitives, replace=False, new_primitives=None):
        self.orig_primitives = orig_primitives
        self.replace = replace
        self.new_primitives = new_primitives


