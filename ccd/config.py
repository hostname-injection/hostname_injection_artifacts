from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class EncoderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    max_length: int = 253
    fp16: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncoderConfig":
        return cls(**data)


@dataclass
class ConeConfig:
    dim: int = 384
    num_cones: int = 4096
    active_cones: int = 8
    temperature: float = 10.0
    axis_init: str = "random"  # random | kmeans
    seed: int = 13

    # LSH parameters (optional)
    use_lsh: bool = True
    lsh_tables: int = 8
    lsh_bits: int = 12
    lsh_probe_radius: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConeConfig":
        return cls(**data)


@dataclass
class PriorConfig:
    smoothing: float = 1e-6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriorConfig":
        return cls(**data)


@dataclass
class CalibrationConfig:
    alpha: float = 1e-4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationConfig":
        return cls(**data)


@dataclass
class ScoringConfig:
    effective_count: float = 1.0
    mixture_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effective_count": float(self.effective_count),
            "mixture_weights": {str(k): float(v) for k, v in self.mixture_weights.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoringConfig":
        weights = data.get("mixture_weights", {})
        return cls(
            effective_count=float(data.get("effective_count", 1.0)),
            mixture_weights={str(k): float(v) for k, v in weights.items()},
        )


@dataclass
class CCDConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    cone: ConeConfig = field(default_factory=ConeConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.to_dict(),
            "cone": self.cone.to_dict(),
            "prior": self.prior.to_dict(),
            "calibration": self.calibration.to_dict(),
            "scoring": self.scoring.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CCDConfig":
        return cls(
            encoder=EncoderConfig.from_dict(data.get("encoder", {})),
            cone=ConeConfig.from_dict(data.get("cone", {})),
            prior=PriorConfig.from_dict(data.get("prior", {})),
            calibration=CalibrationConfig.from_dict(data.get("calibration", {})),
            scoring=ScoringConfig.from_dict(data.get("scoring", {})),
        )
