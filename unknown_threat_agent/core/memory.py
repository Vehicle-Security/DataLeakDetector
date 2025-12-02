"""
共享记忆层 - Memory
基于图数据库的持久化记忆系统，存储实体、关系和属性
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import json
import logging
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """
    图数据库中的节点
    表示一个实体（用户、文件、应用、网站等）
    """
    node_id: str  # 节点唯一标识
    node_type: str  # 节点类型: User, File, App, Website, Process等
    properties: Dict[str, Any] = field(default_factory=dict)  # 节点属性
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.node_id == other.node_id


@dataclass
class Relationship:
    """
    图数据库中的边（关系）
    表示两个实体之间的关系
    """
    source_id: str  # 源节点ID
    target_id: str  # 目标节点ID
    rel_type: str  # 关系类型: created, accessed, uploaded, converted等
    properties: Dict[str, Any] = field(default_factory=dict)  # 关系属性
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __repr__(self):
        return f"({self.source_id})-[{self.rel_type}]->({self.target_id})"


class Memory:
    """
    Agent的记忆系统 - 持久化图数据库
    
    核心功能：
    1. 存储和查询节点（实体）
    2. 存储和查询关系（边）
    3. 支持污点标记和传播
    4. 支持假设（hypothesis）的记录和更新
    5. 路径查询（用于威胁链重建）
    6. 持久化到磁盘
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.nodes: Dict[str, Node] = {}  # node_id -> Node
        self.relationships: List[Relationship] = []  # 所有关系
        self.adjacency_list: Dict[str, Set[str]] = {}  # node_id -> {connected_node_ids}
        self.db_path = db_path
        self._lock = threading.RLock()
        
        if db_path:
            self._load_from_disk()
        
        logger.info(f"记忆系统已初始化，包含 {len(self.nodes)} 个节点")
    
    def add_node(self, node: Node) -> Node:
        """
        添加或更新一个节点
        
        Args:
            node: Node对象
            
        Returns:
            添加/更新后的Node对象
        """
        with self._lock:
            if node.node_id in self.nodes:
                # 更新现有节点
                existing = self.nodes[node.node_id]
                existing.properties.update(node.properties)
                existing.updated_at = datetime.now()
                logger.debug(f"已更新节点: {node.node_id}")
                return existing
            else:
                # 添加新节点
                self.nodes[node.node_id] = node
                self.adjacency_list[node.node_id] = set()
                logger.debug(f"已添加新节点: {node.node_id} ({node.node_type})")
                return node
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """
        获取指定节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            Node对象或None
        """
        return self.nodes.get(node_id)
    
    def update_node_property(self, node_id: str, key: str, value: Any):
        """
        更新节点的某个属性
        
        Args:
            node_id: 节点ID
            key: 属性键
            value: 属性值
        """
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].properties[key] = value
                self.nodes[node_id].updated_at = datetime.now()
                logger.debug(f"已更新 {node_id}.{key} = {value}")
            else:
                logger.warning(f"未找到节点 {node_id}，无法更新属性")
    
    def add_relationship(self, relationship: Relationship):
        """
        添加一个关系（边）
        
        Args:
            relationship: Relationship对象
        """
        with self._lock:
            self.relationships.append(relationship)
            
            # 更新邻接表
            if relationship.source_id not in self.adjacency_list:
                self.adjacency_list[relationship.source_id] = set()
            if relationship.target_id not in self.adjacency_list:
                self.adjacency_list[relationship.target_id] = set()
            
            self.adjacency_list[relationship.source_id].add(relationship.target_id)
            
            logger.debug(f"已添加关系: {relationship}")
    
    def query_nodes(
        self, 
        node_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> List[Node]:
        """
        查询符合条件的节点
        
        Args:
            node_type: 节点类型筛选
            properties: 属性筛选（键值对）
            
        Returns:
            符合条件的节点列表
        """
        with self._lock:
            results = list(self.nodes.values())
        
        if node_type:
            results = [n for n in results if n.node_type == node_type]
        
        if properties:
            for key, value in properties.items():
                results = [n for n in results if n.properties.get(key) == value]
        
        return results
    
    def check_taint(self, node_id: str) -> bool:
        """
        检查节点是否被污点标记
        
        Args:
            node_id: 节点ID
            
        Returns:
            True如果节点被标记为污点，否则False
        """
        node = self.get_node(node_id)
        if node:
            return node.properties.get("is_tainted", False)
        return False
    
    def mark_tainted(self, node_id: str, taint_source: Optional[str] = None):
        """
        标记节点为污点
        
        Args:
            node_id: 节点ID
            taint_source: 污点来源（可选）
        """
        self.update_node_property(node_id, "is_tainted", True)
        if taint_source:
            self.update_node_property(node_id, "taint_source", taint_source)
        logger.info(f"已将 {node_id} 标记为污点 (来源: {taint_source})")
    
    def propagate_taint(self, source_id: str, target_id: str, operation: str):
        """
        污点传播：从源节点传播到目标节点
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            operation: 操作类型（如 "copy", "convert", "derive"）
        """
        if self.check_taint(source_id):
            self.mark_tainted(target_id, taint_source=source_id)
            
            # 添加污点传播关系
            self.add_relationship(Relationship(
                source_id=source_id,
                target_id=target_id,
                rel_type=f"taint_propagation_{operation}",
                properties={"propagated": True}
            ))
            
            logger.info(f"污点已传播: {source_id} -> {target_id} 通过 {operation}")
    
    def set_hypothesis(self, node_id: str, hypothesis: str):
        """
        为节点设置假设（N3推理用）
        
        Args:
            node_id: 节点ID
            hypothesis: 假设内容（如 "Modal_Jump_Tool"）
        """
        self.update_node_property(node_id, "hypothesis", hypothesis)
        logger.info(f"已为 {node_id} 设置假设: {hypothesis}")
    
    def find_paths(
        self, 
        start_id: str, 
        end_id: str, 
        max_depth: int = 5
    ) -> List[List[str]]:
        """
        查找从起点到终点的所有路径（威胁链重建）
        
        Args:
            start_id: 起始节点ID
            end_id: 终止节点ID
            max_depth: 最大路径深度
            
        Returns:
            路径列表，每个路径是节点ID列表
        """
        def dfs(current: str, target: str, path: List[str], visited: Set[str], depth: int):
            if depth > max_depth:
                return []
            
            if current == target:
                return [path]
            
            all_paths = []
            visited.add(current)
            
            for neighbor in self.adjacency_list.get(current, []):
                if neighbor not in visited:
                    paths = dfs(neighbor, target, path + [neighbor], visited, depth + 1)
                    all_paths.extend(paths)
            
            visited.remove(current)
            return all_paths
        
        with self._lock:
            return dfs(start_id, end_id, [start_id], set(), 0)
    
    def get_taint_chain(self, node_id: str) -> List[Node]:
        """
        获取节点的完整污点传播链
        
        Args:
            node_id: 节点ID
            
        Returns:
            污点链上的所有节点（从最初源到当前节点）
        """
        chain = []
        current_id = node_id
        visited = set()
        
        with self._lock:
            while current_id and current_id not in visited:
                visited.add(current_id)
                node = self.get_node(current_id)
                if node:
                    chain.append(node)
                    current_id = node.properties.get("taint_source")
                else:
                    break
        
        return list(reversed(chain))  # 从源到当前节点
    
    def save_to_disk(self):
        """将内存数据持久化到磁盘"""
        if not self.db_path:
            return
        
        data = {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "properties": n.properties,
                    "created_at": n.created_at.isoformat(),
                    "updated_at": n.updated_at.isoformat()
                }
                for n in self.nodes.values()
            ],
            "relationships": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "rel_type": r.rel_type,
                    "properties": r.properties,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.relationships
            ]
        }
        
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"记忆已保存到 {self.db_path}")
    
    def _load_from_disk(self):
        """从磁盘加载数据"""
        if not Path(self.db_path).exists():
            logger.info(f"在 {self.db_path} 未找到现有记忆数据")
            return
        
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载节点
            for node_data in data.get("nodes", []):
                node = Node(
                    node_id=node_data["node_id"],
                    node_type=node_data["node_type"],
                    properties=node_data["properties"],
                    created_at=datetime.fromisoformat(node_data["created_at"]),
                    updated_at=datetime.fromisoformat(node_data["updated_at"])
                )
                self.add_node(node)
            
            # 加载关系
            for rel_data in data.get("relationships", []):
                rel = Relationship(
                    source_id=rel_data["source_id"],
                    target_id=rel_data["target_id"],
                    rel_type=rel_data["rel_type"],
                    properties=rel_data["properties"],
                    timestamp=datetime.fromisoformat(rel_data["timestamp"])
                )
                self.add_relationship(rel)
            
            logger.info(f"已从 {self.db_path} 加载记忆: {len(self.nodes)} 个节点, {len(self.relationships)} 个关系")
        
        except Exception as e:
            logger.error(f"从 {self.db_path} 加载记忆失败: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计信息"""
        with self._lock:
            node_types = {}
            tainted_count = 0
            
            for node in self.nodes.values():
                node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
                if node.properties.get("is_tainted", False):
                    tainted_count += 1
            
            return {
                "total_nodes": len(self.nodes),
                "total_relationships": len(self.relationships),
                "node_type_distribution": node_types,
                "tainted_nodes": tainted_count
            }
