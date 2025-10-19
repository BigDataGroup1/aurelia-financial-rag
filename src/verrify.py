"""
AURELIA - Output Verification & Visualization Script
=====================================================
Run this after processing to:
1. Check all generated files
2. Create visualizations
3. Generate summary report
4. Verify everything worked correctly
"""

import json
from pathlib import Path
from collections import Counter
import statistics

# Visualization (optional but recommended)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    sns.set_style("whitegrid")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  matplotlib/seaborn not installed. Install for visualizations:")
    print("   pip install matplotlib seaborn")


class OutputVerifier:
    """Verify and visualize all outputs"""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.figures_dir = Path("figures")
        self.figures_dir.mkdir(exist_ok=True)
    
    def check_files(self):
        """Check if all expected files exist"""
        print("\n" + "="*70)
        print("📋 CHECKING OUTPUT FILES")
        print("="*70 + "\n")
        
        expected_files = {
            'sections.json': 'Document structure',
            'elements.json': 'Multi-modal elements',
            'chunks.json': 'Semantic chunks'
        }
        
        all_exist = True
        for filename, description in expected_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size / 1024  # KB
                print(f"✅ {filename:20s} ({size:8.1f} KB) - {description}")
            else:
                print(f"❌ {filename:20s} - MISSING!")
                all_exist = False
        
        # Check ChromaDB
        chroma_path = Path("data/chroma_db")
        if chroma_path.exists():
            print(f"✅ {'chroma_db/':20s} - Vector database")
        else:
            print(f"⚠️  {'chroma_db/':20s} - Not found (may use Pinecone)")
        
        return all_exist
    
    def load_data(self):
        """Load all processed data"""
        print("\n📂 Loading data...")
        
        with open(self.data_dir / 'sections.json') as f:
            self.sections = json.load(f)
        
        with open(self.data_dir / 'elements.json') as f:
            self.elements = json.load(f)
        
        with open(self.data_dir / 'chunks.json') as f:
            self.chunks = json.load(f)
        
        print(f"   ✓ Loaded {len(self.sections)} sections")
        print(f"   ✓ Loaded {len(self.elements)} elements")
        print(f"   ✓ Loaded {len(self.chunks)} chunks")
    
    def generate_statistics(self):
        """Generate detailed statistics"""
        print("\n" + "="*70)
        print("📊 DETAILED STATISTICS")
        print("="*70 + "\n")
        
        # Sections
        print("📚 Document Structure:")
        level_counts = Counter([s['level'] for s in self.sections])
        for level in sorted(level_counts.keys()):
            print(f"   Level {level}: {level_counts[level]:4d} sections")
        
        # Elements
        print(f"\n🎨 Multi-Modal Elements (Total: {len(self.elements)}):")
        element_types = Counter([e['element_type'] for e in self.elements])
        for elem_type, count in element_types.most_common():
            print(f"   {elem_type.title():15s}: {count:4d}")
        
        # Chunks
        print(f"\n✂️  Semantic Chunks (Total: {len(self.chunks)}):")
        
        lengths = [c['metadata']['word_count'] for c in self.chunks]
        print(f"   Length Statistics:")
        print(f"      Mean:   {statistics.mean(lengths):6.1f} words")
        print(f"      Median: {statistics.median(lengths):6.1f} words")
        print(f"      Min:    {min(lengths):6d} words")
        print(f"      Max:    {max(lengths):6d} words")
        print(f"      StdDev: {statistics.stdev(lengths):6.1f} words")
        
        chunk_types = Counter([c['chunk_type'] for c in self.chunks])
        print(f"\n   Type Distribution:")
        for ctype, count in chunk_types.most_common():
            pct = (count / len(self.chunks)) * 100
            print(f"      {ctype.title():15s}: {count:4d} ({pct:5.1f}%)")
        
        section_levels = Counter([c['metadata']['section_level'] for c in self.chunks])
        print(f"\n   By Section Level:")
        for level in sorted(section_levels.keys()):
            count = section_levels[level]
            pct = (count / len(self.chunks)) * 100
            print(f"      Level {level}: {count:4d} ({pct:5.1f}%)")
    
    def show_examples(self):
        """Show example chunks"""
        print("\n" + "="*70)
        print("📝 EXAMPLE CHUNKS")
        print("="*70 + "\n")
        
        # Show one of each type
        types_shown = set()
        
        for chunk in self.chunks[:50]:  # Check first 50
            ctype = chunk['chunk_type']
            if ctype not in types_shown:
                print(f"🏷️  Type: {ctype.upper()}")
                print(f"   Section: {chunk['section_title']}")
                print(f"   Length: {chunk['metadata']['word_count']} words")
                print(f"   Content preview:")
                print(f"   {chunk['content'][:200]}...")
                print()
                types_shown.add(ctype)
                
                if len(types_shown) >= 3:  # Show 3 examples
                    break
    
    def create_visualizations(self):
        """Create visualization plots"""
        if not PLOTTING_AVAILABLE:
            print("\n⚠️  Skipping visualizations (matplotlib not installed)")
            return
        
        print("\n" + "="*70)
        print("🎨 GENERATING VISUALIZATIONS")
        print("="*70 + "\n")
        
        # 1. Chunk Length Distribution
        self._plot_chunk_lengths()
        
        # 2. Chunk Type Distribution
        self._plot_chunk_types()
        
        # 3. Section Hierarchy
        self._plot_section_hierarchy()
        
        # 4. Multi-Modal Elements
        self._plot_elements()
        
        print("\n✅ All visualizations saved to figures/")
    
    def _plot_chunk_lengths(self):
        """Plot chunk length distribution"""
        lengths = [c['metadata']['word_count'] for c in self.chunks]
        
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        plt.axvline(statistics.mean(lengths), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {statistics.mean(lengths):.0f}')
        plt.axvline(statistics.median(lengths), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {statistics.median(lengths):.0f}')
        plt.xlabel('Chunk Length (words)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Chunk Lengths', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = self.figures_dir / 'chunk_lengths.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {filepath}")
        plt.close()
    
    def _plot_chunk_types(self):
        """Plot chunk type distribution"""
        types = Counter([c['chunk_type'] for c in self.chunks])
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set3(range(len(types)))
        bars = plt.bar(types.keys(), types.values(), color=colors, 
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Chunk Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Distribution of Chunk Types', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filepath = self.figures_dir / 'chunk_types.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {filepath}")
        plt.close()
    
    def _plot_section_hierarchy(self):
        """Plot section hierarchy"""
        level_counts = Counter([s['level'] for s in self.sections])
        
        plt.figure(figsize=(8, 6))
        levels = sorted(level_counts.keys())
        counts = [level_counts[l] for l in levels]
        colors = ['#e74c3c', '#3498db', '#2ecc71'][:len(levels)]
        
        bars = plt.barh([f'Level {l}' for l in levels], counts, 
                       color=colors, edgecolor='black', linewidth=1.5)
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {int(width)}',
                    ha='left', va='center', fontsize=11, fontweight='bold')
        
        plt.xlabel('Number of Sections', fontsize=12)
        plt.title('Document Hierarchy Structure', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        filepath = self.figures_dir / 'section_hierarchy.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {filepath}")
        plt.close()
    
    def _plot_elements(self):
        """Plot multi-modal elements"""
        element_types = Counter([e['element_type'] for e in self.elements])
        
        if not element_types:
            print("   ⚠️  No multi-modal elements to plot")
            return
        
        plt.figure(figsize=(8, 8))
        colors = {'formula': '#e74c3c', 'code': '#3498db', 
                 'figure': '#2ecc71', 'table': '#f39c12'}
        element_colors = [colors.get(t, '#95a5a6') for t in element_types.keys()]
        
        wedges, texts, autotexts = plt.pie(
            element_types.values(), 
            labels=element_types.keys(),
            autopct='%1.1f%%',
            colors=element_colors,
            startangle=90,
            explode=[0.05]*len(element_types),
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        
        plt.title('Multi-Modal Element Distribution', 
                 fontsize=14, fontweight='bold', pad=20)
        
        filepath = self.figures_dir / 'multimodal_elements.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {filepath}")
        plt.close()
    
    def generate_report(self):
        """Generate markdown report"""
        print("\n" + "="*70)
        print("📄 GENERATING REPORT")
        print("="*70 + "\n")
        
        report = "# AURELIA Lab 1 - Processing Report\n\n"
        report += f"**Generated**: {Path.cwd()}\n\n"
        
        report += "## Summary\n\n"
        report += f"- **Sections Extracted**: {len(self.sections)}\n"
        report += f"- **Multi-Modal Elements**: {len(self.elements)}\n"
        report += f"- **Semantic Chunks**: {len(self.chunks)}\n\n"
        
        # Chunks detail
        lengths = [c['metadata']['word_count'] for c in self.chunks]
        report += "## Chunk Statistics\n\n"
        report += f"- **Average Length**: {statistics.mean(lengths):.1f} words\n"
        report += f"- **Median Length**: {statistics.median(lengths):.1f} words\n"
        report += f"- **Range**: {min(lengths)} - {max(lengths)} words\n\n"
        
        # Types
        chunk_types = Counter([c['chunk_type'] for c in self.chunks])
        report += "## Chunk Type Distribution\n\n"
        for ctype, count in chunk_types.most_common():
            pct = (count / len(self.chunks)) * 100
            report += f"- **{ctype.title()}**: {count} ({pct:.1f}%)\n"
        
        # Elements
        element_types = Counter([e['element_type'] for e in self.elements])
        report += "\n## Multi-Modal Elements\n\n"
        for etype, count in element_types.most_common():
            report += f"- **{etype.title()}**: {count}\n"
        
        # Visualizations
        report += "\n## Generated Visualizations\n\n"
        for fig in self.figures_dir.glob("*.png"):
            report += f"- `{fig.name}`\n"
        
        # Save report
        report_path = self.data_dir / "PROCESSING_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved: {report_path}")
        
        return report
    
    def run_full_verification(self):
        """Run complete verification"""
        print("\n" + "="*70)
        print("🔍 AURELIA LAB 1 - OUTPUT VERIFICATION")
        print("="*70)
        
        # Check files exist
        if not self.check_files():
            print("\n❌ Some files are missing! Run the main script first.")
            return False
        
        # Load data
        self.load_data()
        
        # Generate stats
        self.generate_statistics()
        
        # Show examples
        self.show_examples()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        # Final summary
        print("\n" + "="*70)
        print("✅ VERIFICATION COMPLETE!")
        print("="*70)
        print("\n📁 Check these locations:")
        print(f"   - Data: {self.data_dir}/")
        print(f"   - Figures: {self.figures_dir}/")
        print(f"   - Report: {self.data_dir}/PROCESSING_REPORT.md")
        print("\n💡 Open the PNG files in figures/ to see visualizations!")
        
        return True


def main():
    """Run verification"""
    verifier = OutputVerifier()
    verifier.run_full_verification()


if __name__ == "__main__":
    main()