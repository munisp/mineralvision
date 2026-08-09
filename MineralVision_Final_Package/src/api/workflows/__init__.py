"""
MineralVision Workflows Module.

This module provides end-to-end reference workflows:
- Gold exploration workflows (orogenic, epithermal, intrusion-related, IOCG)
- Lithium exploration workflows (pegmatite, brine, clay-hosted)
- REE exploration workflows (carbonatite, ion-adsorption)
"""

from .reference_workflows import (
    WorkflowStage,
    CommodityType,
    DepositModel,
    WorkflowConfig,
    WorkflowInput,
    StageResult,
    WorkflowResult,
    IngestStage,
    QCStage,
    ProcessingStage,
    InterpretationStage,
    TargetingStage,
    ReportingStage,
    ReferenceWorkflow,
    create_gold_orogenic_workflow,
    create_gold_epithermal_workflow,
    create_lithium_pegmatite_workflow,
    create_lithium_brine_workflow,
    create_ree_carbonatite_workflow,
)

__all__ = [
    # Enums
    'WorkflowStage',
    'CommodityType',
    'DepositModel',
    
    # Data classes
    'WorkflowConfig',
    'WorkflowInput',
    'StageResult',
    'WorkflowResult',
    
    # Stage executors
    'IngestStage',
    'QCStage',
    'ProcessingStage',
    'InterpretationStage',
    'TargetingStage',
    'ReportingStage',
    
    # Main workflow class
    'ReferenceWorkflow',
    
    # Factory functions
    'create_gold_orogenic_workflow',
    'create_gold_epithermal_workflow',
    'create_lithium_pegmatite_workflow',
    'create_lithium_brine_workflow',
    'create_ree_carbonatite_workflow',
]
