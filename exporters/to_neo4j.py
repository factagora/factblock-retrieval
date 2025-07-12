#!/usr/bin/env python3
"""
Neo4j 익스포터

FactBlock 지식 그래프를 Neo4j 데이터베이스로 익스포트
"""

import sys
import os
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase, Driver
import logging
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exporters.neo4j_loader import Neo4jDataLoader

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchLoader:
    """간단한 배치 로더 - 여러 노드 타입을 한 번에 처리"""
    
    def __init__(self, driver, database: str = "neo4j"):
        self.driver = driver
        self.database = database
        self.batch_data = {}
    
    def add_nodes(self, node_type: str, nodes: List[Dict[str, Any]]):
        """배치에 노드 추가"""
        if node_type not in self.batch_data:
            self.batch_data[node_type] = []
        self.batch_data[node_type].extend(nodes)
    
    def execute_batch(self) -> Dict[str, int]:
        """배치 실행 - 모든 노드 타입을 한 트랜잭션에서 처리"""
        results = {}
        
        try:
            with self.driver.session(database=self.database) as session:
                with session.begin_transaction() as tx:
                    for node_type, nodes in self.batch_data.items():
                        if not nodes:
                            continue
                            
                        query = self._get_query_for_type(node_type)
                        if query:
                            tx.run(query, nodes=nodes)
                            results[node_type] = len(nodes)
                            logger.info(f"✅ 배치: {len(nodes)}개 {node_type} 노드 생성")
            
            logger.info(f"🚀 배치 로딩 완료: {sum(results.values())}개 노드")
            return results
            
        except Exception as e:
            logger.error(f"❌ 배치 로딩 실패: {e}")
            return {}
    
    def _get_query_for_type(self, node_type: str) -> str:
        """노드 타입별 Cypher 쿼리 반환"""
        queries = {
            "FactBlock": """
                UNWIND $nodes AS fb
                CREATE (f:FactBlock {
                    id: fb.id, claim: fb.claim, evidence: fb.evidence,
                    verdict: fb.verdict, confidence_score: fb.confidence_score,
                    summary: fb.summary, processed_date: fb.processed_date,
                    status: fb.status, version: fb.version, language: fb.language,
                    source_url: fb.source_url, source_type: fb.source_type,
                    author: fb.author, publication: fb.publication,
                    published_date: fb.published_date, credibility_score: fb.credibility_score,
                    impact_level: fb.impact_level, affected_sectors: fb.affected_sectors,
                    time_horizon: fb.time_horizon, volatility_impact: fb.volatility_impact,
                    investment_themes: fb.investment_themes, alpha_potential: fb.alpha_potential,
                    applicable_strategies: fb.applicable_strategies, risk_factors: fb.risk_factors,
                    total_connections: fb.total_connections, outgoing_count: fb.outgoing_count,
                    incoming_count: fb.incoming_count, created_at: timestamp()
                })""",
            "Entity": """
                UNWIND $nodes AS entity
                CREATE (e:Entity {
                    name: entity.name, entity_type: entity.entity_type,
                    confidence: entity.confidence, factblock_count: entity.factblock_count,
                    factblock_ids: entity.factblock_ids, created_at: timestamp()
                })""",
            "Topic": """
                UNWIND $nodes AS topic
                CREATE (t:Topic {
                    name: topic.name, topic_type: topic.topic_type,
                    relevance_score: topic.relevance_score, factblock_count: topic.factblock_count,
                    factblock_ids: topic.factblock_ids, created_at: timestamp()
                })"""
        }
        return queries.get(node_type, "")

class Neo4jExporter:
    """Neo4j 데이터베이스 익스포터"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Args:
            uri: Neo4j 연결 URI (예: bolt://localhost:7687)
            username: 사용자명
            password: 비밀번호
            database: 데이터베이스명 (기본값: neo4j)
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
        
    def connect(self) -> bool:
        """Neo4j 데이터베이스 연결"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            
            # 연결 테스트
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                
            if test_value == 1:
                logger.info(f"✅ Neo4j 연결 성공: {self.uri}")
                return True
            else:
                logger.error("❌ Neo4j 연결 테스트 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ Neo4j 연결 실패: {e}")
            return False
    
    def close(self):
        """Neo4j 연결 종료"""
        if self.driver:
            self.driver.close()
            logger.info("🔐 Neo4j 연결 종료")
    
    def clear_database(self) -> bool:
        """데이터베이스 초기화 (모든 노드와 관계 삭제)"""
        try:
            with self.driver.session(database=self.database) as session:
                # 모든 노드와 관계 삭제
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("🗑️ 데이터베이스 초기화 완료")
                return True
                
        except Exception as e:
            logger.error(f"❌ 데이터베이스 초기화 실패: {e}")
            return False
    
    def create_indexes(self) -> bool:
        """성능 최적화를 위한 인덱스 생성"""
        indexes = [
            # FactBlock 인덱스
            "CREATE INDEX factblock_id IF NOT EXISTS FOR (f:FactBlock) ON (f.id)",
            "CREATE INDEX factblock_verdict IF NOT EXISTS FOR (f:FactBlock) ON (f.verdict)",
            "CREATE INDEX factblock_impact IF NOT EXISTS FOR (f:FactBlock) ON (f.impact_level)",
            
            # Entity 인덱스
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            
            # Topic 인덱스
            "CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name)",
            "CREATE INDEX topic_type IF NOT EXISTS FOR (t:Topic) ON (t.topic_type)"
        ]
        
        try:
            with self.driver.session(database=self.database) as session:
                for index_query in indexes:
                    session.run(index_query)
                    
            logger.info("📊 인덱스 생성 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌인덱스 생성 실패: {e}")
            return False
    
    def create_factblock_nodes(self, factblocks: List[Dict[str, Any]]) -> int:
        """FactBlock 노드 생성"""
        
        create_query = """
        UNWIND $factblocks AS fb
        CREATE (f:FactBlock {
            id: fb.id,
            claim: fb.claim,
            evidence: fb.evidence,
            verdict: fb.verdict,
            confidence_score: fb.confidence_score,
            summary: fb.summary,
            processed_date: fb.processed_date,
            status: fb.status,
            version: fb.version,
            language: fb.language,
            
            // Source metadata
            source_url: fb.source_url,
            source_type: fb.source_type,
            author: fb.author,
            publication: fb.publication,
            published_date: fb.published_date,
            credibility_score: fb.credibility_score,
            
            // Investment metadata
            impact_level: fb.impact_level,
            affected_sectors: fb.affected_sectors,
            time_horizon: fb.time_horizon,
            volatility_impact: fb.volatility_impact,
            investment_themes: fb.investment_themes,
            alpha_potential: fb.alpha_potential,
            applicable_strategies: fb.applicable_strategies,
            risk_factors: fb.risk_factors,
            
            // Connection stats
            total_connections: fb.total_connections,
            outgoing_count: fb.outgoing_count,
            incoming_count: fb.incoming_count,
            
            created_at: timestamp()
        })
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(create_query, factblocks=factblocks)
                
            logger.info(f"✅ {len(factblocks)}개 FactBlock 노드 생성 완료")
            return len(factblocks)
            
        except Exception as e:
            logger.error(f"❌ FactBlock 노드 생성 실패: {e}")
            return 0
    
    def create_entity_nodes(self, entities: List[Dict[str, Any]]) -> int:
        """Entity 노드 생성"""
        
        create_query = """
        UNWIND $entities AS entity
        CREATE (e:Entity {
            name: entity.name,
            entity_type: entity.entity_type,
            confidence: entity.confidence,
            factblock_count: entity.factblock_count,
            factblock_ids: entity.factblock_ids,
            created_at: timestamp()
        })
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(create_query, entities=entities)
                
            logger.info(f"✅ {len(entities)}개 Entity 노드 생성 완료")
            return len(entities)
            
        except Exception as e:
            logger.error(f"❌ Entity 노드 생성 실패: {e}")
            return 0
    
    def create_topic_nodes(self, topics: List[Dict[str, Any]]) -> int:
        """Topic 노드 생성"""
        
        create_query = """
        UNWIND $topics AS topic
        CREATE (t:Topic {
            name: topic.name,
            topic_type: topic.topic_type,
            relevance_score: topic.relevance_score,
            factblock_count: topic.factblock_count,
            factblock_ids: topic.factblock_ids,
            created_at: timestamp()
        })
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(create_query, topics=topics)
                
            logger.info(f"✅ {len(topics)}개 Topic 노드 생성 완료")
            return len(topics)
            
        except Exception as e:
            logger.error(f"❌ Topic 노드 생성 실패: {e}")
            return 0
    
    def create_factblock_relationships(self, relationships: List[Dict[str, Any]]) -> int:
        """FactBlock 간 관계 생성"""
        
        create_query = """
        UNWIND $relationships AS rel
        MATCH (source:FactBlock {id: rel.source_id})
        MATCH (target:FactBlock {id: rel.target_id})
        CREATE (source)-[r:RELATES_TO {
            id: rel.id,
            relationship_type: rel.relationship_type,
            strength: rel.strength,
            confidence: rel.confidence,
            investment_insight: rel.investment_insight,
            created_date: rel.created_date,
            created_at: timestamp()
        }]->(target)
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(create_query, relationships=relationships)
                
            logger.info(f"✅ {len(relationships)}개 FactBlock 관계 생성 완료")
            return len(relationships)
            
        except Exception as e:
            logger.error(f"❌ FactBlock 관계 생성 실패: {e}")
            return 0
    
    def create_entity_relationships(self, factblocks: List[Dict[str, Any]]) -> int:
        """FactBlock과 Entity 간 관계 생성"""
        
        # 먼저 각 FactBlock의 엔티티 정보를 수집
        entity_relations = []
        
        # Neo4j 데이터 로더로부터 원본 데이터 다시 읽기
        loader = Neo4jDataLoader("data/processed/enhanced_knowledge_graph_dataset.json")
        loader.load_dataset()
        
        for fb_data in loader.raw_data['factblocks']:
            fb_id = fb_data['id']
            for entity_data in fb_data.get('entities', []):
                entity_relations.append({
                    'factblock_id': fb_id,
                    'entity_name': entity_data['name'],
                    'entity_type': entity_data['entity_type'],
                    'confidence': entity_data.get('confidence', 0.0)
                })
        
        create_query = """
        UNWIND $relations AS rel
        MATCH (f:FactBlock {id: rel.factblock_id})
        MATCH (e:Entity {name: rel.entity_name, entity_type: rel.entity_type})
        CREATE (f)-[r:MENTIONS {
            confidence: rel.confidence,
            created_at: timestamp()
        }]->(e)
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(create_query, relations=entity_relations)
                
            logger.info(f"✅ {len(entity_relations)}개 FactBlock-Entity 관계 생성 완료")
            return len(entity_relations)
            
        except Exception as e:
            logger.error(f"❌ FactBlock-Entity 관계 생성 실패: {e}")
            return 0
    
    def create_nodes_batch(self, factblocks: List[Dict[str, Any]], entities: List[Dict[str, Any]], topics: List[Dict[str, Any]]) -> Dict[str, int]:
        """배치 로더를 사용하여 모든 노드를 한 번에 생성"""
        batch_loader = BatchLoader(self.driver, self.database)
        
        # 배치에 노드들 추가
        if factblocks:
            batch_loader.add_nodes("FactBlock", factblocks)
        if entities:
            batch_loader.add_nodes("Entity", entities)
        if topics:
            batch_loader.add_nodes("Topic", topics)
        
        # 배치 실행
        return batch_loader.execute_batch()
    
    def create_topic_relationships(self, factblocks: List[Dict[str, Any]]) -> int:
        """FactBlock과 Topic 간 관계 생성"""
        
        # 각 FactBlock의 토픽 정보를 수집
        topic_relations = []
        
        # Neo4j 데이터 로더로부터 원본 데이터 다시 읽기
        loader = Neo4jDataLoader("data/processed/enhanced_knowledge_graph_dataset.json")
        loader.load_dataset()
        
        for fb_data in loader.raw_data['factblocks']:
            fb_id = fb_data['id']
            for topic_data in fb_data.get('topics', []):
                topic_relations.append({
                    'factblock_id': fb_id,
                    'topic_name': topic_data['name'],
                    'topic_type': topic_data.get('topic_type', 'general'),
                    'relevance_score': topic_data.get('relevance_score', 0.0)
                })
        
        create_query = """
        UNWIND $relations AS rel
        MATCH (f:FactBlock {id: rel.factblock_id})
        MATCH (t:Topic {name: rel.topic_name, topic_type: rel.topic_type})
        CREATE (f)-[r:TAGGED_AS {
            relevance_score: rel.relevance_score,
            created_at: timestamp()
        }]->(t)
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(create_query, relations=topic_relations)
                
            logger.info(f"✅ {len(topic_relations)}개 FactBlock-Topic 관계 생성 완료")
            return len(topic_relations)
            
        except Exception as e:
            logger.error(f"❌ FactBlock-Topic 관계 생성 실패: {e}")
            return 0
    
    def get_database_stats(self) -> Dict[str, int]:
        """데이터베이스 통계 조회"""
        
        stats_queries = {
            'factblocks': "MATCH (f:FactBlock) RETURN count(f) as count",
            'entities': "MATCH (e:Entity) RETURN count(e) as count",
            'topics': "MATCH (t:Topic) RETURN count(t) as count",
            'factblock_relationships': "MATCH ()-[r:RELATES_TO]-() RETURN count(r) as count",
            'entity_relationships': "MATCH ()-[r:MENTIONS]-() RETURN count(r) as count",
            'topic_relationships': "MATCH ()-[r:TAGGED_AS]-() RETURN count(r) as count"
        }
        
        stats = {}
        
        try:
            with self.driver.session(database=self.database) as session:
                for stat_name, query in stats_queries.items():
                    result = session.run(query)
                    stats[stat_name] = result.single()['count']
                    
            return stats
            
        except Exception as e:
            logger.error(f"❌ 데이터베이스 통계 조회 실패: {e}")
            return {}
    
    def export_full_dataset(self, dataset_path: str, clear_existing: bool = True, use_batch: bool = True) -> bool:
        """전체 데이터셋을 Neo4j로 익스포트"""
        
        logger.info("🚀 Neo4j 전체 데이터셋 익스포트 시작...")
        
        # 데이터 로드
        loader = Neo4jDataLoader(dataset_path)
        if not loader.load_dataset():
            return False
        
        # 데이터 파싱
        factblocks, relationships, entities, topics = loader.get_neo4j_data()
        
        # 기존 데이터 삭제 (선택적)
        if clear_existing:
            if not self.clear_database():
                return False
        
        # 인덱스 생성
        if not self.create_indexes():
            return False
        
        # 노드 생성 (배치 모드 또는 개별 모드)
        if use_batch:
            logger.info("📦 배치 모드로 노드 생성...")
            batch_results = self.create_nodes_batch(factblocks, entities, topics)
            fb_count = batch_results.get("FactBlock", 0)
            entity_count = batch_results.get("Entity", 0)
            topic_count = batch_results.get("Topic", 0)
        else:
            logger.info("🔄 개별 모드로 노드 생성...")
            fb_count = self.create_factblock_nodes(factblocks)
            entity_count = self.create_entity_nodes(entities)
            topic_count = self.create_topic_nodes(topics)
        
        # 관계 생성
        rel_count = self.create_factblock_relationships(relationships)
        entity_rel_count = self.create_entity_relationships(factblocks)
        topic_rel_count = self.create_topic_relationships(factblocks)
        
        # 최종 통계
        stats = self.get_database_stats()
        
        logger.info("✅ Neo4j 데이터셋 익스포트 완료!")
        logger.info("📊 최종 통계:")
        logger.info(f"     • FactBlock 노드: {stats.get('factblocks', 0)}개")
        logger.info(f"     • Entity 노드: {stats.get('entities', 0)}개")
        logger.info(f"     • Topic 노드: {stats.get('topics', 0)}개")
        logger.info(f"     • FactBlock 관계: {stats.get('factblock_relationships', 0)}개")
        logger.info(f"     • Entity 관계: {stats.get('entity_relationships', 0)}개")
        logger.info(f"     • Topic 관계: {stats.get('topic_relationships', 0)}개")
        
        return True


def demo_neo4j_export():
    """Neo4j 익스포트 데모 (실제 연결 없이 구조 테스트)"""
    
    print("🧪 Neo4j 익스포터 구조 테스트...")
    
    # 익스포터 초기화 (연결하지 않음)
    exporter = Neo4jExporter(
        uri="bolt://localhost:7687",
        username="neo4j", 
        password="password"
    )
    
    # 데이터 로더 테스트
    dataset_path = "data/processed/enhanced_knowledge_graph_dataset.json"
    loader = Neo4jDataLoader(dataset_path)
    
    if loader.load_dataset():
        factblocks, relationships, entities, topics = loader.get_neo4j_data()
        
        print("✅ 익스포트 준비 완료:")
        print(f"   • FactBlock: {len(factblocks)}개")
        print(f"   • 관계: {len(relationships)}개") 
        print(f"   • 엔티티: {len(entities)}개")
        print(f"   • 토픽: {len(topics)}개")
        
        print("\n💡 실제 Neo4j 연결이 있다면 다음 명령으로 익스포트:")
        print("   exporter.connect()")
        print("   exporter.export_full_dataset(dataset_path)")
        print("   exporter.close()")
        
        return True
    else:
        print("❌ 데이터 로드 실패")
        return False


if __name__ == "__main__":
    demo_neo4j_export()