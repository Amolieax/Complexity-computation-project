import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

class TSPVisualizer:
    """Класс для визуализации результатов тестов TSP"""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        sns.set_palette("husl")
        self.figures = []

    def plot_comparative_analysis(self, test_results, save_path=None):
        """Строит сравнительный анализ всех тестов"""
        n_tests = len(test_results)
        fig = plt.figure(figsize=(16, 4 * ((n_tests + 3) // 4)))
        fig.suptitle('Сравнительный анализ тестов алгоритма Бермана-Карпинского', 
                    fontsize=16, fontweight='bold')
        
        n_cols = min(4, n_tests)
        n_rows = (n_tests + n_cols - 1) // n_cols
        
        for i, results in enumerate(test_results):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            self._plot_single_test(ax, results, i)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def _plot_single_test(self, ax, results, test_idx):
        """Визуализация одного теста"""
        test_name = results['test_name']

        metrics = ['Время (с)', 'Стоимость', 'Доля веса 1']
        values = [
            results['time'],
            results['cost'],
            results['weight_1_ratio']
        ]
        
        bars = ax.bar(metrics, values, alpha=0.8)
        ax.set_title(f"{test_idx+1}. {test_name[:30]}...", fontsize=10)
        ax.set_ylabel('Значение')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if metrics[bars.index(bar)] == 'Время (с)':
                text = f'{val:.3f}'
            elif metrics[bars.index(bar)] == 'Доля веса 1':
                text = f'{val:.1%}'
            else:
                text = f'{val:.1f}'

            ax.text(bar.get_x() + bar.get_width() / 2., height * 1.02,
                   text, ha='center', va='bottom', fontsize=8)

        ax.text(0.02, 0.98, f"n={results['n']}", 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def plot_performance_analysis(self, performance_results, save_path=None):
        """Анализ производительности"""
        sizes = [r['size'] for r in performance_results]
        avg_times = [r['avg_time'] for r in performance_results]
        avg_costs = [r['avg_cost'] for r in performance_results]
        avg_ratios = [r['avg_ratio'] for r in performance_results]
        std_times = [r['std_time'] for r in performance_results]
        std_costs = [r['std_cost'] for r in performance_results]
        
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.errorbar(sizes, avg_times, yerr=std_times, 
                    fmt='o-', linewidth=2, capsize=5, markersize=8,
                    color='royalblue', ecolor='lightblue', elinewidth=2)
        ax1.set_title('Зависимость времени от размера графа')
        ax1.set_xlabel('Размер графа (n)')
        ax1.set_ylabel('Время, с')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.errorbar(sizes, avg_costs, yerr=std_costs,
                    fmt='s-', linewidth=2, capsize=5, markersize=8,
                    color='crimson', ecolor='lightcoral', elinewidth=2)
        ax2.set_title('Зависимость стоимости от размера графа')
        ax2.set_xlabel('Размер графа (n)')
        ax2.set_ylabel('Стоимость тура')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(sizes, avg_ratios, '^-', linewidth=2, markersize=8, color='seagreen')
        ax3.set_title('Доля ребер стоимости 1')
        ax3.set_xlabel('Размер графа (n)')
        ax3.set_ylabel('Доля ребер веса 1')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 0])
        theoretical = [avg_times[0] * (n / sizes[0])**6 for n in sizes]
        
        ax4.plot(sizes, avg_times, 'o-', linewidth=2, markersize=8, 
                label='Фактическое', color='royalblue')
        ax4.plot(sizes, theoretical, '--', linewidth=2, 
                label='Теоретическое O(n⁶)', color='gray')
        ax4.set_title('Сравнение с теоретической сложностью')
        ax4.set_xlabel('Размер графа (n)')
        ax4.set_ylabel('Время, с (log scale)')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3, which='both')
        
        ax5 = fig.add_subplot(gs[1, 1])
        time_data = []
        cost_data = []
        labels = []
        
        for result in performance_results:
            for test in result['tests']:
                time_data.append(test['time'])
                cost_data.append(test['cost'])
                labels.append(f"n={result['size']}")
        
        ax5.scatter(time_data, cost_data, c=[len(l) for l in labels], 
                             cmap='viridis', alpha=0.6, s=100)
        ax5.set_title('Распределение времени и стоимости')
        ax5.set_xlabel('Время, с')
        ax5.set_ylabel('Стоимость')
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 2])
        width = 0.25
        x = np.arange(len(sizes))
        
        norm_times = [t / max(avg_times) for t in avg_times]
        norm_costs = [c / max(avg_costs) for c in avg_costs]
        
        ax6.bar(x - width, norm_times, width, label='Время (норм.)', alpha=0.7)
        ax6.bar(x, norm_costs, width, label='Стоимость (норм.)', alpha=0.7)
        ax6.bar(x + width, avg_ratios, width, label='Доля веса 1', alpha=0.7)
        
        ax6.set_title('Нормализованные метрики')
        ax6.set_xlabel('Размер графа')
        ax6.set_ylabel('Нормализованное значение')
        ax6.set_xticks(x)
        ax6.set_xticklabels(sizes)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Анализ производительности алгоритма', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_edge_distribution(self, test_results, save_path=None):
        """Визуализация распределения ребер по весам"""
        fig = plt.figure(figsize=(6, 5))
        
        test_names = [r['test_name'][:20] + '...' for r in test_results]
        weight_1_counts = [r['weight_1_count'] for r in test_results]
        weight_2_counts = [r['weight_2_count'] for r in test_results]
        
        x = np.arange(len(test_results))
        width = 0.35
        
        plt.bar(x - width / 2, weight_1_counts, width, label='Вес 1', color='green', alpha=0.7)
        plt.bar(x + width / 2, weight_2_counts, width, label='Вес 2', color='red', alpha=0.7)
        plt.title('Распределение ребер по весам')
        plt.xlabel('Тест')
        plt.ylabel('Количество ребер')
        plt.xticks(x, test_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
    
        plt.suptitle('Анализ распределения ребер в турах', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        self.figures.append(fig)

    def show_all_figures(self):
        """Показывает все сохраненные графики"""
        plt.show()
    
    def save_all_figures(self, prefix="tsp_analysis"):
        """Сохраняет все графики в файлы"""
        for i, fig in enumerate(self.figures):
            fig.savefig(f"{prefix}_{i:02d}.png", dpi=300, bbox_inches='tight')