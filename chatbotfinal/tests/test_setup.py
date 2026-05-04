import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import *

def test_setup():
    print("=" * 50)
    print("STATOUR Multi-Agent Setup Verification")
    print("=" * 50)
    
    # Check API Key
    if OPENAI_API_KEY and OPENAI_API_KEY != "sk-your-key-here":
        print("✅ OpenAI API Key configured")
    else:
        print("❌ OpenAI API Key NOT configured — update .env file")
    
    # Check Tavily Key
    if TAVILY_API_KEY and TAVILY_API_KEY != "tvly-your-key-here":
        print("✅ Tavily API Key configured")
    else:
        print("⚠️  Tavily API Key NOT configured — needed for Researcher Agent")
    
    # Check APF Data
    if os.path.exists(APF_DATA_PATH):
        import pandas as pd
        df = pd.read_excel(APF_DATA_PATH)
        print(f"✅ APF Data loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df['date_stat'].min()} to {df['date_stat'].max()}")
        print(f"   Nationalities: {df['nationalite'].nunique()}")
        print(f"   Border posts: {df['poste_frontiere'].nunique()}")
        print(f"   Regions: {df['region'].nunique()}")
    else:
        print(f"❌ APF Data NOT found at {APF_DATA_PATH}")
        print(f"   Please place your APF file in the data/ folder")
    
    # Check directories
    for dir_name, dir_path in [("Data", DATA_DIR), ("Knowledge Base", KNOWLEDGE_BASE_DIR), 
                                 ("Documents", DOCUMENTS_DIR), ("VectorStore", VECTORSTORE_DIR)]:
        if os.path.exists(dir_path):
            print(f"✅ {dir_name} directory exists")
        else:
            print(f"❌ {dir_name} directory missing: {dir_path}")
    
    print("=" * 50)
    print(f"Model: {OPENAI_MODEL}")
    print("=" * 50)

if __name__ == "__main__":
    test_setup()