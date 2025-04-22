"""
Index management for vector database.
"""
import os
import logging
import pickle
import shutil
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

from config import VECTOR_STORE_CONFIG

logger = logging.getLogger(__name__)

class IndexManager:
    """Manager for vector index operations like backup and optimization."""
    
    def __init__(self, index_name: Optional[str] = None):
        """
        Initialize the index manager.
        
        Args:
            index_name: Optional custom index name
        """
        self.index_name = index_name or VECTOR_STORE_CONFIG["index_name"]
        self.index_path = f"data/indexes/{self.index_name}.pkl"
        self.backup_dir = f"data/backups/{self.index_name}"
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        logger.info(f"Index manager initialized for index: {self.index_name}")
    
    def create_backup(self, tag: Optional[str] = None) -> str:
        """
        Create a backup of the current index.
        
        Args:
            tag: Optional tag for the backup
            
        Returns:
            Path to the backup file
        """
        if not os.path.exists(self.index_path):
            logger.error("Cannot backup: Index file does not exist")
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_suffix = f"_{tag}" if tag else ""
        backup_filename = f"{self.index_name}_{timestamp}{tag_suffix}.pkl"
        backup_path = os.path.join(self.backup_dir, backup_filename)
        
        try:
            # Copy the index file to backup location
            shutil.copy2(self.index_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            raise
    
    def restore_from_backup(self, backup_path: Optional[str] = None) -> bool:
        """
        Restore index from a backup.
        
        Args:
            backup_path: Path to specific backup file, or None to use latest
            
        Returns:
            True if restore was successful
        """
        # If no specific backup specified, find latest
        if not backup_path:
            backup_files = [f for f in os.listdir(self.backup_dir) 
                           if f.startswith(self.index_name) and f.endswith('.pkl')]
            
            if not backup_files:
                logger.error("No backups found to restore")
                return False
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda f: os.path.getmtime(
                os.path.join(self.backup_dir, f)), reverse=True)
            
            backup_path = os.path.join(self.backup_dir, backup_files[0])
        
        try:
            # Create backup of current index before restore
            if os.path.exists(self.index_path):
                self.create_backup(tag="pre_restore")
            
            # Copy backup to active index location
            shutil.copy2(backup_path, self.index_path)
            logger.info(f"Restored index from backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            return False
    
    def optimize_index(self, vector_store) -> bool:
        """
        Optimize the index for better performance.
        
        Args:
            vector_store: VectorStore instance to optimize
            
        Returns:
            True if optimization was successful
        """
        try:
            # Create backup before optimization
            self.create_backup(tag="pre_optimize")
            
            # Specific optimization depends on the vector store type
            # For FAISS, we can train the index for better quantization
            if hasattr(vector_store.vector_store, 'index') and hasattr(vector_store.vector_store.index, 'train'):
                # Get all vectors
                vectors = []
                for doc_id in vector_store.vector_store.docstore._dict:
                    doc = vector_store.vector_store.docstore.search(doc_id)
                    if hasattr(doc, 'embedding'):
                        vectors.append(doc.embedding)
                
                if vectors:
                    # Convert to numpy array
                    vectors_array = np.array(vectors).astype('float32')
                    
                    # Train the index
                    vector_store.vector_store.index.train(vectors_array)
                    
                    # Save optimized index
                    with open(self.index_path, "wb") as f:
                        pickle.dump(vector_store.vector_store, f)
                    
                    logger.info(f"Optimized index with {len(vectors)} vectors")
                    return True
            
            logger.info("Index optimization not applicable for this vector store type")
            return False
            
        except Exception as e:
            logger.error(f"Index optimization failed: {str(e)}")
            return False
    
    def get_index_stats(self, vector_store) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Args:
            vector_store: VectorStore instance
            
        Returns:
            Dictionary with index statistics
        """
        stats = {
            "index_name": self.index_name,
            "index_file": self.index_path,
            "exists": os.path.exists(self.index_path),
            "size_bytes": 0,
            "last_modified": None,
            "vector_count": 0,
            "dimension": VECTOR_STORE_CONFIG["dimension"],
            "backup_count": 0
        }
        
        # Get file stats if exists
        if stats["exists"]:
            stats["size_bytes"] = os.path.getsize(self.index_path)
            stats["last_modified"] = datetime.fromtimestamp(
                os.path.getmtime(self.index_path)).isoformat()
        
        # Get vector count
        if hasattr(vector_store, 'vector_store') and hasattr(vector_store.vector_store, 'docstore'):
            stats["vector_count"] = len(vector_store.vector_store.docstore._dict)
        
        # Count backups
        if os.path.exists(self.backup_dir):
            stats["backup_count"] = len([
                f for f in os.listdir(self.backup_dir)
                if f.startswith(self.index_name) and f.endswith('.pkl')
            ])
        
        return stats