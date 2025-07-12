# Neo4j FactBlock Database Analysis Report

## Executive Summary

I successfully analyzed your Neo4j FactBlock database at `bolt://localhost:7687` and extracted concrete, factual claims to create realistic example texts for GraphRAG testing. The database contains **100 FactBlocks** with **29 containing concrete numerical/financial data** that will provide excellent test cases for fact-checking.

## Database Structure Overview

### Database Statistics
- **Nodes**: 122 total
- **Relationships**: 250 total
- **FactBlocks**: 100

### Node Types
- **FactBlock**: 100 (main content nodes)
- **Entity**: 10 (companies/organizations)
- **Topic, Company, Fund, Investor, Deal, Security**: Various counts

### Relationship Types
- **MENTIONS**: 100 (FactBlocks → Entities)
- **TAGGED_AS**: 100 (FactBlocks → Topics)
- **RELATES_TO**: 50 (FactBlocks → FactBlocks)

## Key Entities in Database

The following entities are mentioned across FactBlocks:
1. **OPEC** - Multiple references in energy sector
2. **연준 (Federal Reserve)** - Central banking references
3. **현대자동차 (Hyundai Motor)** - Automotive industry
4. **OpenAI** - Technology sector
5. **Palantir** - Software/AI sector
6. **유럽연합 (European Union)** - Regulatory policies
7. **VinFast** - Electric vehicles
8. **보령제약** - Pharmaceuticals
9. **Market** - General market references
10. **정부 (Government)** - Policy references

## Sector Coverage Analysis

### Most Covered Sectors (by FactBlock count):
1. **Semiconductors**: 19 FactBlocks
2. **Healthcare**: 18 FactBlocks  
3. **Technology**: 10 FactBlocks
4. **Automobiles**: 9 FactBlocks
5. **Financials**: 8 FactBlocks
6. **Pharmaceuticals**: 8 FactBlocks
7. **Energy**: 7 FactBlocks
8. **Transportation**: 7 FactBlocks
9. **Software**: 6 FactBlocks
10. **Renewable Energy**: 5 FactBlocks

## Concrete Claims with Financial/Numerical Data

### Top High-Confidence Claims:

#### 1. EU Internal Combustion Engine Ban (Confidence: 0.95)
- **Claim**: "유럽연합이 2035년 내 내연기관차 판매 금지를 선언했다"
- **Evidence**: "EU가 탄소 중립 목표 달성을 위해 2035년부터 신규 내연기관 자동차 판매를 전면 금지한다고 발표했다"
- **Concrete Facts**: 2035년, 내연기관차 전면 금지
- **Sector**: Automobiles

#### 2. OPEC Oil Production Cut (Confidence: 0.94)
- **Claim**: "OPEC이 감산 합의에 도달했다"
- **Evidence**: "주요 산유국들이 원유 생산량을 일일 200만 배럴 감축하기로 합의했다"
- **Concrete Facts**: 200만 배럴, 일일 감축
- **Sector**: Energy

#### 3. Airline Fuel Cost Increase (Confidence: 0.94)
- **Claim**: "글로벌 항공사의 연료비가 15% 상승했다"
- **Evidence**: "원유 공급 감소로 인한 유가 상승이 항공유 가격을 직접적으로 끌어올렸다"
- **Concrete Facts**: 15% 상승
- **Sector**: Transportation

#### 4. Semiconductor Shortage Timeline (Confidence: 0.94)
- **Claim**: "반도체 부족 현상이 2024년까지 지속될 전망이다"
- **Evidence**: "코로나19 여파와 지정학적 긴장으로 인한 반도체 공급망 차질이 장기화되고 있다"
- **Concrete Facts**: 2024년까지
- **Sector**: Semiconductors

#### 5. Hyundai Production Cut (Confidence: 0.94)
- **Claim**: "현대자동차는 차량 생산량을 15% 감축한다고 발표했다"
- **Evidence**: "반도체 수급 불안정으로 인해 현대자동차가 주요 차종의 생산 일정을 조정하고 출하량을 줄이기로 결정했다"
- **Concrete Facts**: 15% 감축
- **Sector**: Automobiles

#### 6. Federal Reserve Interest Rate Hikes (Confidence: 0.93)
- **Claim**: "미국 연준이 2022년 기준금리를 7차례 인상했다"
- **Evidence**: "연방준비제도가 인플레이션 억제를 위해 공격적인 통화긴축 정책을 실시했다"
- **Concrete Facts**: 2022년, 7차례, 기준금리 인상
- **Sector**: Financials

## Cross-Reference Relationships

### Documented Causal Chains:

#### 1. Energy → Transportation Chain
- **OPEC oil production cuts** → **Airline fuel cost increases**
- Relationship: `RELATES_TO` 
- Cross-sector impact from energy policy to transportation costs

#### 2. Monetary Policy → Banking Chain  
- **Federal Reserve rate hikes** → **Bank lending demand decline**
- Relationship: `RELATES_TO`
- Policy transmission mechanism through financial sector

#### 3. Technology → Semiconductor Chain
- **AI model releases (GPT-5, Llama-3)** → **Edge computing chip demand**
- Relationship: `RELATES_TO`
- Technology advancement driving hardware requirements

#### 4. Semiconductor → Automotive Chain
- **Semiconductor shortage** → **Hyundai production cuts**
- Relationship: Supply chain impact
- Manufacturing bottlenecks affecting production

## Realistic Test Examples Generated

Based on the actual database content, I created **5 realistic fact-check examples**:

### Example 1: OPEC Energy Impact
```
Text: "OPEC이 감산 합의에 도달했으며, 주요 산유국들이 원유 생산량을 일일 200만 배럴 감축하기로 합의했다"
Expected Evidence: Multiple FactBlocks about oil production cuts and energy market impacts
Concrete Facts: 200만 배럴, 일일 감축
```

### Example 2: Airline Fuel Costs
```
Text: "글로벌 항공사의 연료비가 15% 상승했으며, 원유 공급 감소로 인한 유가 상승이 항공유 가격을 직접적으로 끌어올렸다"
Expected Evidence: Cross-sector relationship between energy and transportation
Concrete Facts: 15% 상승
```

### Example 3: Federal Reserve Policy
```
Text: "미국 연준이 2022년 기준금리를 7차례 인상했으며, 연방준비제도가 인플레이션 억제를 위해 공격적인 통화긴축 정책을 실시했다"
Expected Evidence: Multiple FactBlocks about monetary policy and financial sector impacts
Concrete Facts: 2022년, 7차례, 기준금리 인상
```

### Example 4: Semiconductor Supply Chain
```
Text: "반도체 부족 현상이 2024년까지 지속될 전망이다 코로나19 여파와 지정학적 긴장으로 인한 반도체 공급망 차질이 장기화되고 있다"
Expected Evidence: Supply chain analysis and timeline projections
Concrete Facts: 2024년까지
```

### Example 5: Automotive Production Impact
```
Text: "현대자동차는 차량 생산량을 15% 감축한다고 발표했다 반도체 수급 불안정으로 인해 현대자동차가 주요 차종의 생산 일정을 조정하고 출하량을 줄이기로 결정했다"
Expected Evidence: Company-specific production decisions linked to supply chain issues
Concrete Facts: 15% 감축
```

## Cross-Reference Examples

I also generated **3 cross-reference examples** that span multiple FactBlocks:

### 1. Energy Sector Chain Reaction
```
Text: "원유 생산량 감축으로 인한 에너지 시장 충격이 항공 업계에도 영향을 미쳤다. OPEC의 감산 합의 이후 항공사들의 연료비가 크게 상승했다."
Cross-references: OPEC production cuts + Airline fuel cost increases
Relationship Type: Causal chain
```

### 2. Federal Reserve Policy Ripple Effects
```
Text: "미국 연준의 공격적인 금리 인상이 글로벌 시장에 광범위한 영향을 미쳤다. 2022년 7차례 금리 인상은 다양한 산업 부문의 투자 심리를 위축시켰다."
Cross-references: Federal Reserve policy + Multiple sector impacts
Relationship Type: Policy impact
```

### 3. Semiconductor-Automotive Supply Chain
```
Text: "반도체 부족 현상이 자동차 산업에 직접적인 타격을 주었다. 현대자동차가 생산량을 15% 감축한 것도 반도체 수급 문제 때문이다."
Cross-references: Semiconductor shortage + Hyundai production cuts
Relationship Type: Supply chain impact
```

## Key Insights for GraphRAG Testing

### 1. **Rich Numerical Data**
Your database contains excellent concrete facts:
- Specific percentages (15% increases/decreases)
- Large quantities (200만 배럴)
- Specific dates and timelines (2022년, 2024년까지, 2035년)
- Frequency data (7차례)

### 2. **Cross-Sector Relationships**
Strong documented relationships between:
- Energy → Transportation (oil prices → fuel costs)
- Semiconductors → Automotive (chip shortage → production cuts)
- Financial → Banking (Fed policy → lending)
- Technology → Hardware (AI models → chip demand)

### 3. **High-Confidence Claims**
Most claims have confidence scores > 0.89, indicating reliable evidence backing

### 4. **Real Entity Coverage**
Major corporations and organizations are well-represented:
- OPEC, Federal Reserve, Hyundai, OpenAI, Palantir, EU

### 5. **Multi-Language Content**
Claims are in Korean with English entity names, providing good test coverage for multilingual fact-checking

## Recommendations for GraphRAG Testing

### Use These Realistic Examples Instead of Hypothetical Ones:

1. **Test energy sector claims** using OPEC oil production data
2. **Test financial sector claims** using Federal Reserve policy data  
3. **Test automotive claims** using Hyundai production cuts
4. **Test cross-sector relationships** using energy→transportation chains
5. **Test entity recognition** using well-documented companies like OPEC, 연준, 현대자동차

### Files Generated:
- `realistic_factcheck_examples.json` - 5 concrete examples based on actual claims
- `cross_reference_examples.json` - 3 multi-FactBlock relationship examples

These examples will provide much more realistic testing scenarios than hypothetical claims, as they're based on actual data in your GraphRAG system and will return real evidence when fact-checked.