from enum import Enum


class AttackType(Enum):
    ACTOR = "ActorBreaker"
    COA = "ChainOfAttack"
    CRESCENDO = "Crescendo"
    FITD = "FootInTheDoor"
    X_TEAMING = "XTeaming"
    MIX = "Mix" # allows mixing attack components

    INTERACTIVE = "Interactive" # not a real attack; allows interacting with engine through command line
    
