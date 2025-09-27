import argparse

def main():
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser(description='一个简单的问候程序')
    
    # 2. 添加参数
    parser.add_argument('--name', type=str, default='World', help='要问候的人名')
    parser.add_argument('--times', type=int, default=1, help='问候次数')
    parser.add_argument('--uppercase', action='store_true', help='是否使用大写')
    
    # 3. 解析参数
    args = parser.parse_args()

    print(args)
    
    # 4. 使用参数
    greeting = f"Hello, {args.name}!"
    if args.uppercase:
        greeting = greeting.upper()
    
    for i in range(args.times):
        print(greeting)

if __name__ == "__main__":
    main()