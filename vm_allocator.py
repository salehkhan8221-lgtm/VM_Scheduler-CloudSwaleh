"""
Advanced VM Allocation Module
Implements multiple allocation strategies with resource constraints.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """VM allocation strategies."""
    THRESHOLD_BASED = "threshold_based"
    PREDICTIVE = "predictive"
    LOAD_BALANCING = "load_balancing"
    HYBRID = "hybrid"


@dataclass
class VMResource:
    """Virtual Machine resource specification."""
    vm_id: int
    cpu_cores: float
    memory: float  # MB
    storage: float  # GB
    allocated_at: int  # Timestamp
    duration: float
    
    def __repr__(self):
        return f"VM{self.vm_id}(CPU:{self.cpu_cores}core,MEM:{self.memory}MB,STOR:{self.storage}GB)"


@dataclass
class HostResource:
    """Physical host resource specification."""
    host_id: int
    total_cpu_cores: float
    total_memory: float  # MB
    total_storage: float  # GB
    allocated_cpu: float = 0
    allocated_memory: float = 0
    allocated_storage: float = 0
    vms: List[int] = None
    
    def __post_init__(self):
        if self.vms is None:
            self.vms = []
    
    @property
    def available_cpu(self) -> float:
        return self.total_cpu_cores - self.allocated_cpu
    
    @property
    def available_memory(self) -> float:
        return self.total_memory - self.allocated_memory
    
    @property
    def available_storage(self) -> float:
        return self.total_storage - self.allocated_storage
    
    @property
    def cpu_utilization(self) -> float:
        return (self.allocated_cpu / self.total_cpu_cores * 100) if self.total_cpu_cores > 0 else 0
    
    @property
    def memory_utilization(self) -> float:
        return (self.allocated_memory / self.total_memory * 100) if self.total_memory > 0 else 0
    
    @property
    def storage_utilization(self) -> float:
        return (self.allocated_storage / self.total_storage * 100) if self.total_storage > 0 else 0
    
    def can_allocate(self, vm: VMResource) -> bool:
        """Check if VM can be allocated to this host."""
        return (self.available_cpu >= vm.cpu_cores and
                self.available_memory >= vm.memory and
                self.available_storage >= vm.storage)
    
    def allocate(self, vm: VMResource) -> bool:
        """Allocate VM to this host."""
        if not self.can_allocate(vm):
            return False
        
        self.allocated_cpu += vm.cpu_cores
        self.allocated_memory += vm.memory
        self.allocated_storage += vm.storage
        self.vms.append(vm.vm_id)
        
        return True
    
    def deallocate(self, vm: VMResource) -> bool:
        """Deallocate VM from this host."""
        if vm.vm_id not in self.vms:
            return False
        
        self.allocated_cpu = max(0, self.allocated_cpu - vm.cpu_cores)
        self.allocated_memory = max(0, self.allocated_memory - vm.memory)
        self.allocated_storage = max(0, self.allocated_storage - vm.storage)
        self.vms.remove(vm.vm_id)
        
        return True
    
    def __repr__(self):
        return (f"Host{self.host_id}(CPU:{self.cpu_utilization:.1f}%,"
                f"MEM:{self.memory_utilization:.1f}%,VMs:{len(self.vms)})")


class VMAllocator:
    """Advanced VM allocation with multiple strategies."""
    
    def __init__(self, num_hosts: int = 10, cpu_per_host: float = 8.0,
                 memory_per_host: float = 16384, storage_per_host: float = 500):
        """
        Initialize allocator with datacenter resources.
        
        Args:
            num_hosts: Number of physical hosts
            cpu_per_host: CPU cores per host
            memory_per_host: Memory (MB) per host
            storage_per_host: Storage (GB) per host
        """
        self.hosts = [
            HostResource(i, cpu_per_host, memory_per_host, storage_per_host)
            for i in range(num_hosts)
        ]
        
        self.allocation_history = []
        self.vm_counter = 0
        self.cpu_threshold = 80
        self.memory_threshold = 85
        self.storage_threshold = 80
    
    def create_vm(self, cpu: float = 1.0, memory: float = 1024,
                 storage: float = 10, duration: float = 10) -> VMResource:
        """
        Create a new VM resource.
        
        Args:
            cpu: CPU cores needed
            memory: Memory (MB) needed
            storage: Storage (GB) needed
            duration: Expected duration
            
        Returns:
            VMResource instance
        """
        vm = VMResource(
            vm_id=self.vm_counter,
            cpu_cores=cpu,
            memory=memory,
            storage=storage,
            allocated_at=0,
            duration=duration
        )
        self.vm_counter += 1
        return vm
    
    def allocate_threshold_based(self, predicted_cpu: float) -> Tuple[bool, int, str]:
        """
        Allocate VMs based on CPU threshold.
        
        Args:
            predicted_cpu: Predicted CPU usage percentage
            
        Returns:
            Tuple of (success, host_id, message)
        """
        if predicted_cpu > self.cpu_threshold:
            # Create new VM based on prediction
            num_vms = int((predicted_cpu - self.cpu_threshold) / 10) + 1
            
            allocated_count = 0
            for _ in range(num_vms):
                vm = self.create_vm(cpu=0.5, memory=512, storage=5)
                
                for host in self.hosts:
                    if host.can_allocate(vm):
                        if host.allocate(vm):
                            allocated_count += 1
                            self.allocation_history.append({
                                'vm': vm,
                                'host': host.host_id,
                                'timestamp': 0,
                                'strategy': AllocationStrategy.THRESHOLD_BASED.value,
                                'cpu_trigger': predicted_cpu
                            })
                            break
            
            return True, self.hosts[0].host_id, f"Allocated {allocated_count} VMs"
        
        return False, -1, "CPU prediction below threshold"
    
    def allocate_load_balanced(self, vm: VMResource) -> Tuple[bool, int, str]:
        """
        Allocate VM to least loaded host.
        
        Args:
            vm: VM to allocate
            
        Returns:
            Tuple of (success, host_id, message)
        """
        # Find host with lowest average utilization
        best_host = None
        min_utilization = 100
        
        for host in self.hosts:
            if host.can_allocate(vm):
                avg_util = (host.cpu_utilization + host.memory_utilization + 
                           host.storage_utilization) / 3
                
                if avg_util < min_utilization:
                    min_utilization = avg_util
                    best_host = host
        
        if best_host:
            if best_host.allocate(vm):
                self.allocation_history.append({
                    'vm': vm,
                    'host': best_host.host_id,
                    'timestamp': 0,
                    'strategy': AllocationStrategy.LOAD_BALANCING.value
                })
                return True, best_host.host_id, f"Allocated to Host{best_host.host_id}"
        
        return False, -1, "No suitable host found"
    
    def allocate_predictive(self, predicted_values: np.ndarray) -> Dict:
        """
        Predictive allocation based on forecast values.
        
        Args:
            predicted_values: Array of predicted CPU values
            
        Returns:
            Allocation summary dictionary
        """
        allocations = {
            'total_vms': 0,
            'successful': 0,
            'failed': 0,
            'summary': []
        }
        
        for pred_cpu in predicted_values:
            success, host_id, message = self.allocate_threshold_based(pred_cpu)
            
            if success:
                allocations['successful'] += 1
            else:
                allocations['failed'] += 1
            
            allocations['total_vms'] += 1
            allocations['summary'].append({
                'predicted_cpu': pred_cpu,
                'success': success,
                'host_id': host_id,
                'message': message
            })
        
        logger.info(f"Predictive allocation: {allocations['successful']} successful, "
                   f"{allocations['failed']} failed out of {allocations['total_vms']}")
        
        return allocations
    
    def get_host_statistics(self) -> pd.DataFrame:
        """Get statistics for all hosts."""
        stats = []
        
        for host in self.hosts:
            stats.append({
                'host_id': host.host_id,
                'cpu_utilization': host.cpu_utilization,
                'memory_utilization': host.memory_utilization,
                'storage_utilization': host.storage_utilization,
                'vms_count': len(host.vms),
                'available_cpu': host.available_cpu,
                'available_memory': host.available_memory,
                'available_storage': host.available_storage
            })
        
        return pd.DataFrame(stats)
    
    def get_datacenter_utilization(self) -> Dict:
        """Get overall datacenter utilization."""
        total_cpu = sum(h.total_cpu_cores for h in self.hosts)
        allocated_cpu = sum(h.allocated_cpu for h in self.hosts)
        
        total_memory = sum(h.total_memory for h in self.hosts)
        allocated_memory = sum(h.allocated_memory for h in self.hosts)
        
        total_storage = sum(h.total_storage for h in self.hosts)
        allocated_storage = sum(h.allocated_storage for h in self.hosts)
        
        return {
            'cpu_utilization': (allocated_cpu / total_cpu * 100) if total_cpu > 0 else 0,
            'memory_utilization': (allocated_memory / total_memory * 100) if total_memory > 0 else 0,
            'storage_utilization': (allocated_storage / total_storage * 100) if total_storage > 0 else 0,
            'total_vms': sum(len(h.vms) for h in self.hosts),
            'hosts_with_available_resources': sum(1 for h in self.hosts 
                                                  if h.available_cpu > 0 or h.available_memory > 0)
        }
    
    def set_thresholds(self, cpu: float = 80, memory: float = 85, storage: float = 80):
        """Set utilization thresholds."""
        self.cpu_threshold = cpu
        self.memory_threshold = memory
        self.storage_threshold = storage
        logger.info(f"Thresholds set: CPU={cpu}%, Memory={memory}%, Storage={storage}%")
    
    def reset(self):
        """Reset allocator state."""
        for host in self.hosts:
            host.allocated_cpu = 0
            host.allocated_memory = 0
            host.allocated_storage = 0
            host.vms = []
        
        self.allocation_history = []
        self.vm_counter = 0
        logger.info("Allocator reset")
    
    def get_allocation_report(self) -> str:
        """Generate allocation report."""
        util = self.get_datacenter_utilization()
        
        report = "\n" + "="*60 + "\n"
        report += "DATACENTER ALLOCATION REPORT\n"
        report += "="*60 + "\n"
        report += f"Total VMs Allocated: {util['total_vms']}\n"
        report += f"CPU Utilization: {util['cpu_utilization']:.2f}%\n"
        report += f"Memory Utilization: {util['memory_utilization']:.2f}%\n"
        report += f"Storage Utilization: {util['storage_utilization']:.2f}%\n"
        report += f"Hosts with Available Resources: {util['hosts_with_available_resources']}/{len(self.hosts)}\n"
        report += "="*60 + "\n"
        
        return report
