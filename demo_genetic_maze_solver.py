import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# 设置字体避免中文显示问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DemoGeneticMazeSolver:
    """
    遗传算法迷宫求解器演示版本
    按照文章思路实现：个体编码、适应度函数、选择、交叉、变异
    """
    
    def __init__(self):
        # 生成32x32的随机迷宫
        self.maze = self.generate_random_maze(32, 32)
        
        self.start_pos = self.find_position('S')
        self.end_pos = self.find_position('E')
        
        # 移动方向编码 - 核心个体表示方法
        self.directions = {
            'U': (-1, 0),  # 向上
            'D': (1, 0),   # 向下  
            'L': (0, -1),  # 向左
            'R': (0, 1)    # 向右
        }
        
        print("=== 迷宫求解遗传算法演示 ===")
        print(f"起点: {self.start_pos}, 终点: {self.end_pos}")
        print("个体编码: 字符串序列(U/D/L/R表示移动方向)")
        
    def generate_random_maze(self, width, height):
        """生成随机迷宫"""
        # 初始化全墙迷宫
        maze = [[1 for _ in range(width)] for _ in range(height)]
        
        # 使用深度优先搜索生成迷宫
        def carve_path(x, y):
            maze[y][x] = 0  # 将当前位置设为路径
            
            # 定义四个方向：上、右、下、左
            directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
            random.shuffle(directions)
            
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (0 < new_x < width-1 and 0 < new_y < height-1 and 
                    maze[new_y][new_x] == 1):
                    # 打通中间的墙
                    maze[y + dy//2][x + dx//2] = 0
                    carve_path(new_x, new_y)
        
        # 从随机起点开始生成迷宫
        start_x = random.randrange(1, width-1, 2)
        start_y = random.randrange(1, height-1, 2)
        carve_path(start_x, start_y)
        
        # 设置起点和终点
        # 确保起点和终点在路径上
        while True:
            start_x = random.randrange(1, width-1)
            start_y = random.randrange(1, height-1)
            if maze[start_y][start_x] == 0:
                maze[start_y][start_x] = 'S'
                break
        
        while True:
            end_x = random.randrange(1, width-1)
            end_y = random.randrange(1, height-1)
            if maze[end_y][end_x] == 0 and (end_x != start_x or end_y != start_y):
                maze[end_y][end_x] = 'E'
                break
        
        return maze
    
    def find_position(self, target):
        """寻找迷宫中目标位置的坐标"""
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                if self.maze[i][j] == target:
                    return (i, j)
        return None
    
    def generate_initial_population(self, population_size):
        """生成初始种群"""
        population = []
        max_path_length = 200  # 增加最大路径长度
        
        # 使用多种初始化策略
        for i in range(population_size):
            if i < population_size * 0.3:  # 30%使用启发式路径
                path = self.generate_heuristic_path(max_path_length)
            elif i < population_size * 0.7:  # 40%使用随机游走
                path = self.generate_random_walk_path(max_path_length)
            else:  # 30%完全随机
                path = ''.join(random.choices('UDLR', k=random.randint(10, max_path_length)))
            population.append(path)
        return population
    
    def generate_heuristic_path(self, max_length):
        """生成启发式路径"""
        path = []
        current_pos = self.start_pos
        visited = set([current_pos])
        
        while len(path) < max_length:
            # 70%概率向终点方向移动
            if random.random() < 0.7:
                dx = self.end_pos[0] - current_pos[0]
                dy = self.end_pos[1] - current_pos[1]
                
                # 优先选择能减少到终点距离的方向
                possible_moves = []
                if dx > 0 and (current_pos[0] + 1, current_pos[1]) not in visited:
                    possible_moves.append('R')
                if dx < 0 and (current_pos[0] - 1, current_pos[1]) not in visited:
                    possible_moves.append('L')
                if dy > 0 and (current_pos[0], current_pos[1] + 1) not in visited:
                    possible_moves.append('D')
                if dy < 0 and (current_pos[0], current_pos[1] - 1) not in visited:
                    possible_moves.append('U')
                
                if possible_moves:
                    move = random.choice(possible_moves)
                else:
                    move = random.choice('UDLR')
            else:
                move = random.choice('UDLR')
            
            # 尝试移动
            new_pos = self.get_next_position(current_pos, move)
            if self.is_valid_position(new_pos):
                current_pos = new_pos
                visited.add(current_pos)
                path.append(move)
                
                if current_pos == self.end_pos:
                    break
        
        return ''.join(path)
    
    def generate_random_walk_path(self, max_length):
        """生成随机游走路径"""
        path = []
        current_pos = self.start_pos
        visited = set([current_pos])
        
        while len(path) < max_length:
            # 优先选择未访问的位置
            possible_moves = []
            for move in 'UDLR':
                new_pos = self.get_next_position(current_pos, move)
                if self.is_valid_position(new_pos) and new_pos not in visited:
                    possible_moves.append(move)
            
            if possible_moves:
                move = random.choice(possible_moves)
            else:
                move = random.choice('UDLR')
            
            new_pos = self.get_next_position(current_pos, move)
            if self.is_valid_position(new_pos):
                current_pos = new_pos
                visited.add(current_pos)
                path.append(move)
                
                if current_pos == self.end_pos:
                    break
        
        return ''.join(path)
    
    def fitness_function(self, path):
        """改进的适应度函数"""
        current_pos = self.start_pos
        visited = set([current_pos])
        valid_moves = 0
        distance_to_end = float('inf')
        
        # 模拟路径移动
        for move in path:
            new_pos = self.get_next_position(current_pos, move)
            if self.is_valid_position(new_pos):
                current_pos = new_pos
                valid_moves += 1
                visited.add(current_pos)
                
                # 更新到终点的距离
                distance = abs(current_pos[0] - self.end_pos[0]) + abs(current_pos[1] - self.end_pos[1])
                distance_to_end = min(distance_to_end, distance)
        
        # 成功到达终点
        if current_pos == self.end_pos:
            success_reward = 10000  # 增加成功奖励
            efficiency_reward = 1000 * (1 - len(path) / 200)  # 路径效率奖励
            return success_reward + efficiency_reward
        
        # 未到达终点时的适应度
        distance_penalty = distance_to_end * 100  # 距离惩罚
        exploration_reward = len(visited) * 50  # 探索奖励
        path_length_penalty = len(path) * 10  # 路径长度惩罚
        revisit_penalty = (len(path) - len(visited)) * 20  # 重复访问惩罚
        
        return exploration_reward - distance_penalty - path_length_penalty - revisit_penalty
    
    def selection(self, population):
        """
        选择操作 - 根据适应度选择优秀个体作为父代
        使用轮盘赌选择方法
        """
        # 计算所有个体的适应度
        fitness_scores = [self.fitness_function(path) for path in population]
        total_fitness = sum(fitness_scores)
        
        if total_fitness == 0:
            return random.choices(population, k=len(population))
        
        # 计算选择概率 - 适应度越高被选中概率越大
        probabilities = [score / total_fitness for score in fitness_scores]
        
        # 根据概率选择父代
        selected = []
        for _ in range(len(population)):
            selected.append(random.choices(population, weights=probabilities)[0])
        
        return selected
    
    def get_next_position(self, current_pos, move):
        """获取下一个位置"""
        if move in self.directions:
            dx, dy = self.directions[move]
            return (current_pos[0] + dx, current_pos[1] + dy)
        return current_pos

    def is_valid_position(self, pos):
        """检查位置是否有效"""
        if not pos:
            return False
        x, y = pos
        return (0 <= x < len(self.maze) and 
                0 <= y < len(self.maze[0]) and 
                self.maze[x][y] != 1)

    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """锦标赛选择"""
        # 随机选择tournament_size个个体
        tournament_indices = random.sample(range(len(population)), tournament_size)
        # 选择适应度最高的个体
        winner_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[winner_index]

    def crossover(self, parent1, parent2):
        """
        改进的交叉操作 - 使用多种交叉策略
        返回单个子代而不是两个
        """
        if len(parent1) <= 1 or len(parent2) <= 1:
            return parent1
        
        # 随机选择交叉策略
        strategy = random.choice(['single_point', 'two_point', 'uniform'])
        
        if strategy == 'single_point':
            return self.single_point_crossover(parent1, parent2)[0]  # 只返回第一个子代
        elif strategy == 'two_point':
            return self.two_point_crossover(parent1, parent2)[0]  # 只返回第一个子代
        else:
            return self.uniform_crossover(parent1, parent2)[0]  # 只返回第一个子代
    
    def single_point_crossover(self, parent1, parent2):
        """单点交叉"""
        min_length = min(len(parent1), len(parent2))
        if min_length <= 1:
            return parent1, parent2
        
        crossover_point = random.randint(1, min_length - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def two_point_crossover(self, parent1, parent2):
        """两点交叉"""
        min_length = min(len(parent1), len(parent2))
        if min_length <= 2:
            return self.single_point_crossover(parent1, parent2)
        
        point1 = random.randint(1, min_length - 2)
        point2 = random.randint(point1 + 1, min_length - 1)
        
        child1 = (parent1[:point1] + parent2[point1:point2] + 
                 parent1[point2:])
        child2 = (parent2[:point1] + parent1[point1:point2] + 
                 parent2[point2:])
        
        return child1, child2
    
    def uniform_crossover(self, parent1, parent2):
        """均匀交叉"""
        min_length = min(len(parent1), len(parent2))
        if min_length <= 1:
            return parent1, parent2
        
        child1 = []
        child2 = []
        
        for i in range(min_length):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        
        # 处理剩余部分
        if len(parent1) > min_length:
            child1.extend(parent1[min_length:])
        if len(parent2) > min_length:
            child2.extend(parent2[min_length:])
        
        return ''.join(child1), ''.join(child2)
    
    def mutation(self, path, mutation_rate=0.1):
        """
        改进的变异操作 - 使用多种变异策略
        """
        if not path:
            return path
        
        # 随机选择变异策略
        strategy = random.choice(['replace', 'insert', 'delete', 'swap'])
        
        if strategy == 'replace':
            return self.replace_mutation(path, mutation_rate)
        elif strategy == 'insert':
            return self.insert_mutation(path, mutation_rate)
        elif strategy == 'delete':
            return self.delete_mutation(path, mutation_rate)
        else:
            return self.swap_mutation(path, mutation_rate)
    
    def replace_mutation(self, path, mutation_rate):
        """替换变异 - 随机替换某些移动方向"""
        mutated = list(path)
        directions = ['U', 'D', 'L', 'R']
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.choice(directions)
        
        return ''.join(mutated)
    
    def insert_mutation(self, path, mutation_rate):
        """插入变异 - 随机插入新的移动方向"""
        if len(path) >= 100:  # 限制最大长度
            return path
        
        mutated = list(path)
        directions = ['U', 'D', 'L', 'R']
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated.insert(i, random.choice(directions))
                if len(mutated) >= 100:
                    break
        
        return ''.join(mutated)
    
    def delete_mutation(self, path, mutation_rate):
        """删除变异 - 随机删除某些移动方向"""
        if len(path) <= 10:  # 保持最小长度
            return path
        
        mutated = list(path)
        i = 0
        while i < len(mutated):
            if random.random() < mutation_rate:
                mutated.pop(i)
                if len(mutated) <= 10:
                    break
            else:
                i += 1
        
        return ''.join(mutated)
    
    def swap_mutation(self, path, mutation_rate):
        """交换变异 - 随机交换相邻的移动方向"""
        mutated = list(path)
        
        for i in range(len(mutated) - 1):
            if random.random() < mutation_rate:
                mutated[i], mutated[i + 1] = mutated[i + 1], mutated[i]
        
        return ''.join(mutated)
    
    def solve_maze(self, population_size=300, generations=1000, mutation_rate=0.2):
        """改进的求解函数"""
        population = self.generate_initial_population(population_size)
        best_fitness = float('-inf')
        best_solution = None
        generations_without_improvement = 0
        best_history = []
        avg_history = []
        
        for generation in range(generations):
            # 评估种群
            fitness_scores = [self.fitness_function(path) for path in population]
            current_best_fitness = max(fitness_scores)
            current_avg_fitness = sum(fitness_scores) / len(fitness_scores)
            
            # 更新最佳解
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = population[fitness_scores.index(current_best_fitness)]
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # 记录历史
            best_history.append(best_fitness)
            avg_history.append(current_avg_fitness)
            
            # 动态调整变异率
            if generations_without_improvement > 20:
                mutation_rate = min(0.5, mutation_rate * 1.1)
            else:
                mutation_rate = max(0.1, mutation_rate * 0.95)
            
            # 选择精英
            elite_size = int(population_size * 0.1)
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_size]
            new_population = [population[i] for i in elite_indices]
            
            # 生成新一代
            while len(new_population) < population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                child = self.crossover(parent1, parent2)  # 现在只返回一个子代
                child = self.mutation(child, mutation_rate)
                new_population.append(child)
            
            population = new_population
            
            # 如果连续30代没有改进，重新生成部分种群
            if generations_without_improvement > 30:
                population = population[:elite_size] + self.generate_initial_population(population_size - elite_size)
                generations_without_improvement = 0
            
            # 每20代输出一次进度
            if generation % 20 == 0:
                print(f"第 {generation} 代: 最佳适应度 = {best_fitness:.2f}, 平均适应度 = {current_avg_fitness:.2f}")
            
            # 如果找到满意解，提前结束
            if best_fitness >= 10000 and generation > 50:
                print(f"在第 {generation} 代找到满意解！")
                break
        
        return best_solution, best_history, avg_history
    
    def display_solution(self, best_path):
        """
        可视化展示求解结果
        """
        print(f"\n步骤5: 展示求解结果")
        
        if not best_path:
            print("未找到有效路径")
            return
        
        # 创建可视化矩阵
        maze_visual = np.array([[0 if cell == 0 or cell == 'S' or cell == 'E' else 1 
                               for cell in row] for row in self.maze])
        
        # 标记起点和终点
        maze_visual[self.start_pos[0], self.start_pos[1]] = 2  # 起点
        maze_visual[self.end_pos[0], self.end_pos[1]] = 3     # 终点
        
        # 模拟路径并标记
        x, y = self.start_pos
        path_positions = [(x, y)]
        
        print(f"\n路径模拟过程:")
        print(f"起始位置: {(x, y)}")
        
        for i, move in enumerate(best_path):
            if move in self.directions:
                dx, dy = self.directions[move]
                new_x, new_y = x + dx, y + dy
                
                # 检查移动有效性
                if (0 <= new_x < len(self.maze) and 0 <= new_y < len(self.maze[0]) and
                    self.maze[new_x][new_y] != 1):
                    x, y = new_x, new_y
                    path_positions.append((x, y))
                    
                    # 标记路径
                    if (x, y) != self.start_pos and (x, y) != self.end_pos:
                        maze_visual[x, y] = 4
                    
                    print(f"步骤{i+1}: {move} -> {(x, y)}")
                    
                    # 到达终点
                    if (x, y) == self.end_pos:
                        print(" 成功到达终点!")
                        break
                else:
                    print(f"步骤{i+1}: {move} -> 撞墙/越界，停止")
                    break
        
        # 绘制结果
        plt.figure(figsize=(10, 8))
        
        colors = ['white', 'black', 'green', 'red', 'blue']
        cmap = ListedColormap(colors)
        
        plt.imshow(maze_visual, cmap=cmap, interpolation='nearest')
        
        success = "SUCCESS" if path_positions[-1] == self.end_pos else "FAILED"
        plt.title(f'Genetic Algorithm Maze Solver - {success}\n'
                 f'Path: {best_path}\n'
                 f'Steps: {len(path_positions)-1}, Fitness: {self.fitness_function(best_path):.2f}')
        
        plt.grid(True, alpha=0.3)
        
        # 图例
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='white', edgecolor='black', label='Path'),
            plt.Rectangle((0,0),1,1, facecolor='black', label='Wall'),
            plt.Rectangle((0,0),1,1, facecolor='green', label='Start'),
            plt.Rectangle((0,0),1,1, facecolor='red', label='End'),
            plt.Rectangle((0,0),1,1, facecolor='blue', label='Solution Path')
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # 输出详细结果
        print(f"\n=== 最终结果 ===")
        print(f"最优路径编码: {best_path}")
        print(f"路径长度: {len(best_path)}")
        print(f"实际移动步数: {len(path_positions)-1}")
        print(f"是否到达终点: {'是' if path_positions[-1] == self.end_pos else '否'}")
        print(f"最终位置: {path_positions[-1]}")
        print(f"适应度得分: {self.fitness_function(best_path):.2f}")
    
    def plot_evolution_history(self, best_history, avg_history):
        """绘制进化历史曲线"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(best_history, 'r-', label='Best Fitness', linewidth=2)
        plt.plot(avg_history, 'b-', label='Average Fitness', linewidth=1)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # 成功率曲线
        success_threshold = 1000
        success_rate = [(1 if fitness >= success_threshold else 0) for fitness in best_history]
        cumulative_success = np.cumsum(success_rate)
        generations = range(len(best_history))
        
        plt.plot(generations, cumulative_success, 'g-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Cumulative Success Count')
        plt.title('Solution Discovery Progress')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数 - 完整演示遗传算法求解迷宫"""
    print("=== 遗传算法迷宫求解完整演示 ===")
    print("本程序演示了文章中提到的遗传算法核心步骤：")
    print("1. 个体编码（字符串表示路径）")
    print("2. 适应度函数（评估路径质量）") 
    print("3. 选择操作（轮盘赌选择）")
    print("4. 交叉操作（单点交叉）")
    print("5. 变异操作（随机变异）")
    print("6. 迭代进化（多代优化）")
    
    # 创建求解器
    solver = DemoGeneticMazeSolver()
    
    # 显示迷宫
    print(f"\n迷宫布局 ({len(solver.maze)}x{len(solver.maze[0])}):")
    for row in solver.maze:
        print(' '.join(str(cell).center(2) for cell in row))
    
    # 执行遗传算法求解
    best_solution, best_history, avg_history = solver.solve_maze(
        population_size=300,  # 增大种群大小
        generations=1000,     # 增加迭代代数
        mutation_rate=0.2     # 调整变异率
    )
    
    # 展示结果
    solver.display_solution(best_solution)
    solver.plot_evolution_history(best_history, avg_history)
    
    print(f"\n=== 算法原理总结 ===")
    print("1. 初始化：生成随机路径字符串作为初始种群")
    print("2. 评估：使用适应度函数评价每个路径的优劣")
    print("3. 选择：基于适应度选择优秀个体作为父代")
    print("4. 交叉：两个父代交换基因片段产生子代")
    print("5. 变异：随机改变子代的部分基因增加多样性")
    print("6. 迭代：重复步骤2-5直到找到满意解或达到最大代数")
    print("\n遗传算法通过模拟自然进化过程，逐步优化路径质量，")
    print("最终找到从起点到终点的最优或近优路径。")

if __name__ == "__main__":
    main() 