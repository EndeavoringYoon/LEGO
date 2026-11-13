from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

@dataclass
class Joint:
    """One robot joint in world coordinates (A-pose), as read from the MJCF."""
    name: str
    joint_type: str                  # "revolute" | "slide" | "free" | "virtual"
    axis_world: np.ndarray           # (3,)
    position: np.ndarray             # (3,)
    joint_tag: Optional[str] = None  # "roll" | "pitch" | "yaw" | "virtual"
    axis_local: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Keep backward-compatible shape used by existing pipeline."""
        return {
            "position": self.position,
            "type": self.joint_type,
            "joint_tag": self.joint_tag,
            "axis_world": self.axis_world,
        }

@dataclass
class RobotModel:
    """A robot's joints keyed by name, plus the joints left out of the representation."""
    name: str
    joints: Dict[str, Joint] = field(default_factory=dict)
    ignored_joints: List[str] = field(default_factory=list)  # "etc"
    fixed_joints: List[str] = field(default_factory=list)    # "fix"

    def add_joint(self, j: Joint) -> None:
        """Add (or replace) a joint under its name."""
        self.joints[j.name] = j