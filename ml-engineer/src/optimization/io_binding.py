"""
I/O Binding and Pre-Allocated Memory Manager.
Prevents CPU-to-GPU memory transfer bottlenecks by binding contiguous device memory buffers.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import onnxruntime as ort


class FastIOBindingRunner:
    """
    Manages pre-allocated input/output buffers and I/O Binding for an ONNXRuntime session.
    Slashes deep learning inference overhead by eliminating dynamic memory allocation during loops.
    """
    def __init__(self, session: ort.InferenceSession, device_type: str = "cpu", device_id: int = 0):
        self.session = session
        self.device_type = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
        self.device_id = device_id
        self.io_binding = self.session.io_binding()
        self.input_names = [inp.name for inp in session.get_inputs()]
        self.output_names = [out.name for out in session.get_outputs()]

    def run_with_binding(self, feed_dict: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Executes inference using zero-copy / bound I/O tensors.
        """
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()

        # Bind inputs to device memory
        for name, array in feed_dict.items():
            if not array.flags["C_CONTIGUOUS"]:
                array = np.ascontiguousarray(array)

            ort_val = ort.OrtValue.ortvalue_from_numpy(
                array,
                device_type=self.device_type,
                device_id=self.device_id
            )
            self.io_binding.bind_ortvalue_input(name, ort_val)

        # Bind outputs to device memory
        for out_name in self.output_names:
            self.io_binding.bind_output(out_name, device_type=self.device_type, device_id=self.device_id)

        # Synchronous hardware execution
        self.session.run_with_iobinding(self.io_binding)

        # Retrieve outputs
        ort_outputs = self.io_binding.get_outputs()
        return [out.numpy() for out in ort_outputs]
