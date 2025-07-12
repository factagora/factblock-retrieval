#!/usr/bin/env python3
"""
Neo4j 데이터 로더 및 파서

enhanced_knowledge_graph_dataset.json의 FactBlock과 관계 데이터를 
Neo4j 가져오기에 적합한 형태로 로드하고 파싱
"""

import json
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

class Neo4jDataLoader:
    """Enhanced knowledge graph dataset을 Neo4j용으로 로드하고 파싱"""
    
    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: enhanced_knowledge_graph_dataset.json 파일 경로
        """
        self.dataset_path = dataset_path
        self.raw_data = None
        self.factblocks = []
        self.relationships = []
        self.entities = []
        self.topics = []
        
    def load_dataset(self) -> bool:
        """데이터셋 로드"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            
            print(f"✅ 데이터셋 로드 완료:")
            print(f"   • 총 FactBlock: {self.raw_data['metadata']['total_factblocks']}개")
            print(f"   • 총 관계: {self.raw_data['metadata']['total_relationships']}개")
            print(f"   • 버전: {self.raw_data['metadata']['version']}")
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터셋 로드 실패: {e}")
            return False
    
    def parse_factblocks(self) -> List[Dict[str, Any]]:
        """FactBlock을 Neo4j 노드 형태로 파싱"""
        parsed_factblocks = []
        
        for fb_data in self.raw_data['factblocks']:
            try:
                # Neo4j 노드를 위한 기본 속성
                node_props = {
                    'id': fb_data['id'],
                    'claim': fb_data['claim'],
                    'evidence': fb_data['evidence'],
                    'verdict': fb_data['verdict'],
                    'confidence_score': fb_data['confidence_score'],
                    'summary': fb_data.get('summary', ''),
                    'processed_date': fb_data['processed_date'],
                    'status': fb_data['status'],
                    'version': fb_data['version'],
                    'language': fb_data['source_metadata'].get('language', 'ko')
                }
                
                # 소스 메타데이터 추가
                source = fb_data['source_metadata']
                node_props.update({
                    'source_url': source.get('source_url', ''),
                    'source_type': source.get('source_type', ''),
                    'author': source.get('author', ''),
                    'publication': source.get('publication', ''),
                    'published_date': source.get('published_date', ''),
                    'credibility_score': source.get('credibility_score', 0.0)
                })
                
                # 투자 메타데이터 추가
                if 'financial_metadata' in fb_data:
                    fin_meta = fb_data['financial_metadata']
                    
                    # Market impact
                    if 'market_impact' in fin_meta:
                        mi = fin_meta['market_impact']
                        node_props.update({
                            'impact_level': mi.get('impact_level', ''),
                            'affected_sectors': mi.get('affected_sectors', []),
                            'time_horizon': mi.get('time_horizon', ''),
                            'volatility_impact': mi.get('volatility_impact', 0.0)
                        })
                    
                    # Investment themes
                    if 'investment_themes' in fin_meta:
                        themes = fin_meta['investment_themes']
                        node_props['investment_themes'] = [t.get('theme_name', '') for t in themes]
                    
                    # Alpha potential and strategies
                    node_props.update({
                        'alpha_potential': fin_meta.get('alpha_potential', 0.0),
                        'applicable_strategies': fin_meta.get('applicable_strategies', []),
                        'risk_factors': fin_meta.get('risk_factors', [])
                    })
                
                # 연결 통계 추가
                node_props.update({
                    'total_connections': fb_data.get('total_connections', 0),
                    'outgoing_count': len(fb_data.get('outgoing_relationships', [])),
                    'incoming_count': len(fb_data.get('incoming_relationships', []))
                })
                
                parsed_factblocks.append(node_props)
                
            except Exception as e:
                print(f"⚠️ FactBlock 파싱 오류 (ID: {fb_data.get('id', 'unknown')}): {e}")
                continue
        
        self.factblocks = parsed_factblocks
        print(f"✅ {len(parsed_factblocks)}개 FactBlock 파싱 완료")
        return parsed_factblocks
    
    def parse_relationships(self) -> List[Dict[str, Any]]:
        """관계를 Neo4j 엣지 형태로 파싱"""
        parsed_relationships = []
        
        # Inter-factblock relationships 파싱
        if 'inter_factblock_relationships' in self.raw_data:
            for rel_data in self.raw_data['inter_factblock_relationships']:
                try:
                    rel_props = {
                        'id': rel_data['id'],
                        'source_id': rel_data['source_factblock_id'],
                        'target_id': rel_data['target_factblock_id'],
                        'relationship_type': rel_data['relationship_type'],
                        'strength': rel_data['strength'],
                        'confidence': rel_data['confidence'],
                        'investment_insight': rel_data['investment_insight'],
                        'created_date': rel_data.get('created_date', datetime.now().isoformat())
                    }
                    
                    parsed_relationships.append(rel_props)
                    
                except Exception as e:
                    print(f"⚠️ 관계 파싱 오류: {e}")
                    continue
        
        self.relationships = parsed_relationships
        print(f"✅ {len(parsed_relationships)}개 관계 파싱 완료")
        return parsed_relationships
    
    def extract_entities(self) -> List[Dict[str, Any]]:
        """모든 FactBlock에서 고유 엔티티 추출"""
        entity_map = {}
        
        for fb_data in self.raw_data['factblocks']:
            for entity_data in fb_data.get('entities', []):
                entity_name = entity_data['name']
                entity_type = entity_data['entity_type']
                
                # 고유 키 생성 (이름 + 타입)
                entity_key = f"{entity_name}:{entity_type}"
                
                if entity_key not in entity_map:
                    entity_map[entity_key] = {
                        'name': entity_name,
                        'entity_type': entity_type,
                        'confidence': entity_data.get('confidence', 0.0),
                        'factblock_count': 1,
                        'factblock_ids': [fb_data['id']]
                    }
                else:
                    # 이미 존재하는 엔티티면 카운트 증가
                    entity_map[entity_key]['factblock_count'] += 1
                    entity_map[entity_key]['factblock_ids'].append(fb_data['id'])
                    # 신뢰도는 평균으로 업데이트
                    current_conf = entity_map[entity_key]['confidence']
                    new_conf = entity_data.get('confidence', 0.0)
                    entity_map[entity_key]['confidence'] = (current_conf + new_conf) / 2
        
        self.entities = list(entity_map.values())
        print(f"✅ {len(self.entities)}개 고유 엔티티 추출 완료")
        return self.entities
    
    def extract_topics(self) -> List[Dict[str, Any]]:
        """모든 FactBlock에서 고유 토픽 추출"""
        topic_map = {}
        
        for fb_data in self.raw_data['factblocks']:
            for topic_data in fb_data.get('topics', []):
                topic_name = topic_data['name']
                topic_type = topic_data.get('topic_type', 'general')
                
                topic_key = f"{topic_name}:{topic_type}"
                
                if topic_key not in topic_map:
                    topic_map[topic_key] = {
                        'name': topic_name,
                        'topic_type': topic_type,
                        'relevance_score': topic_data.get('relevance_score', 0.0),
                        'factblock_count': 1,
                        'factblock_ids': [fb_data['id']]
                    }
                else:
                    topic_map[topic_key]['factblock_count'] += 1
                    topic_map[topic_key]['factblock_ids'].append(fb_data['id'])
                    # 관련성 점수는 최대값으로 업데이트
                    current_score = topic_map[topic_key]['relevance_score']
                    new_score = topic_data.get('relevance_score', 0.0)
                    topic_map[topic_key]['relevance_score'] = max(current_score, new_score)
        
        self.topics = list(topic_map.values())
        print(f"✅ {len(self.topics)}개 고유 토픽 추출 완료")
        return self.topics
    
    def get_neo4j_data(self) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Neo4j 가져오기용 전체 데이터 반환"""
        if not self.raw_data:
            raise ValueError("데이터를 먼저 로드해야 합니다. load_dataset()을 호출하세요.")
        
        # 모든 데이터 파싱
        factblocks = self.parse_factblocks()
        relationships = self.parse_relationships()
        entities = self.extract_entities()
        topics = self.extract_topics()
        
        return factblocks, relationships, entities, topics
    
    def print_summary(self):
        """데이터 요약 출력"""
        if not self.raw_data:
            print("❌ 로드된 데이터가 없습니다.")
            return
        
        print("\n📊 Neo4j 데이터 준비 요약:")
        print(f"   • FactBlock 노드: {len(self.factblocks)}개")
        print(f"   • 관계 엣지: {len(self.relationships)}개")
        print(f"   • 고유 엔티티: {len(self.entities)}개")
        print(f"   • 고유 토픽: {len(self.topics)}개")
        
        if self.relationships:
            rel_types = {}
            for rel in self.relationships:
                rel_type = rel['relationship_type']
                rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
            
            print("\n📈 관계 유형별 분포:")
            for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
                print(f"   • {rel_type}: {count}개")


def demo_neo4j_loader():
    """Neo4j 데이터 로더 데모"""
    dataset_path = "data/processed/enhanced_knowledge_graph_dataset.json"
    
    print("🚀 Neo4j 데이터 로더 시작...")
    
    # 로더 초기화 및 데이터 로드
    loader = Neo4jDataLoader(dataset_path)
    
    if not loader.load_dataset():
        print("❌ 데이터셋 로드 실패")
        return
    
    # 전체 데이터 파싱
    try:
        factblocks, relationships, entities, topics = loader.get_neo4j_data()
        
        # 요약 출력
        loader.print_summary()
        
        print("\n🔍 샘플 데이터:")
        
        # FactBlock 샘플
        if factblocks:
            sample_fb = factblocks[0]
            print(f"\n📄 샘플 FactBlock:")
            print(f"   ID: {sample_fb['id']}")
            print(f"   Claim: \"{sample_fb['claim'][:50]}...\"")
            print(f"   영향도: {sample_fb.get('impact_level', 'N/A')}")
            print(f"   연결 수: {sample_fb.get('total_connections', 0)}")
        
        # 관계 샘플
        if relationships:
            sample_rel = relationships[0]
            print(f"\n🔗 샘플 관계:")
            print(f"   유형: {sample_rel['relationship_type']}")
            print(f"   신뢰도: {sample_rel['confidence']:.3f}")
            print(f"   인사이트: \"{sample_rel['investment_insight']}\"")
        
        print("\n✅ Neo4j 데이터 로더 완료! 데이터가 준비되었습니다.")
        
        return loader
        
    except Exception as e:
        print(f"❌ 데이터 파싱 중 오류: {e}")
        return None


if __name__ == "__main__":
    demo_neo4j_loader()