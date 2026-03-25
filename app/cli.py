"""
职能沟通翻译助手 - 命令行工具

支持本地命令行使用，方便测试和调试
"""

import sys
import asyncio
import os
from typing import Optional

from app.llm_client import get_llm_client
from app.prompts import (
    get_product_to_dev_prompts,
    get_dev_to_product_prompts,
    get_scene_detection_prompts
)

def print_header():
    """打印标题"""
    print("\n" + "="*60)
    print("  职能沟通翻译助手")
    print("  让产品经理和开发工程师更好地理解彼此")
    print("="*60 + "\n")

def print_usage():
    """打印使用说明"""
    print("使用方式:")
    print("  1. 直接运行程序，通过交互式菜单选择功能")
    print("  2. 命令行参数方式:")
    print("     python -m app.cli --help")
    print("     python -m app.cli --mode product --text '你的输入'")
    print("     python -m app.cli --mode dev --text '你的输入'")
    print("     python -m app.cli --detect --text '你的输入'")
    print("\n")

def print_result(result: str, mode: str):
    """打印翻译结果"""
    if mode == "product_to_dev":
        print("\n📋 【翻译结果 - 开发工程师视角】\n")
    else:
        print("\n⚙️ 【翻译结果 - 产品经理视角】\n")
    
    print(result)
    print("\n" + "-"*60 + "\n")

async def translate_async(text: str, mode: str, use_stream: bool = True) -> str:
    """
    执行翻译
    
    Args:
        text: 待翻译文本
        mode: 翻译模式 (product_to_dev | dev_to_product)
        use_stream: 是否使用流式输出
        
    Returns:
        翻译结果
    """
    llm = get_llm_client()
    
    if mode == "product_to_dev":
        system_prompt, user_prompt = get_product_to_dev_prompts(text)
    else:
        system_prompt, user_prompt = get_dev_to_product_prompts(text)
    
    if use_stream:
        result = []
        async for chunk in llm.stream_chat(system_prompt, user_prompt):
            print(chunk, end="", flush=True)
            result.append(chunk)
        return "".join(result)
    else:
        result = await llm.chat(system_prompt, user_prompt)
        return result

async def detect_scene_async(text: str) -> str:
    """
    检测输入内容的场景
    
    Args:
        text: 待检测文本
        
    Returns:
        场景类型 (product | dev | unknown)
    """
    llm = get_llm_client()
    
    system_prompt, user_prompt = get_scene_detection_prompts(text)
    
    result = await llm.chat(system_prompt, user_prompt, temperature=0, max_tokens=50)
    
    scene = result.strip().lower()
    if scene not in ["product", "dev", "unknown"]:
        scene = "unknown"
    
    return scene

def interactive_mode():
    """交互式模式"""
    print_header()
    print_usage()
    
    while True:
        print("\n请选择功能：")
        print("  1. 产品经理 → 开发工程师（需求翻译为技术方案）")
        print("  2. 开发工程师 → 产品经理（技术方案翻译为业务价值）")
        print("  3. 自动识别输入视角")
        print("  0. 退出\n")
        
        choice = input("请输入选项（0-3）：").strip()
        
        if choice == "0":
            print("\n感谢使用，再见！\n")
            break
        
        if choice == "1":
            mode = "product_to_dev"
            direction_desc = "产品经理 → 开发工程师"
        elif choice == "2":
            mode = "dev_to_product"
            direction_desc = "开发工程师 → 产品经理"
        elif choice == "3":
            text = input("\n请输入要检测的内容：\n> ").strip()
            if not text:
                print("输入不能为空")
                continue
            print("\n正在检测...\n")
            scene = asyncio.run(detect_scene_async(text))
            if scene == "product":
                print(f"✅ 检测结果：产品经理视角")
                print(f"   建议翻译方向：产品经理 → 开发工程师")
            elif scene == "dev":
                print(f"✅ 检测结果：开发工程师视角")
                print(f"   建议翻译方向：开发工程师 → 产品经理")
            else:
                print(f"⚠️ 无法确定视角，请手动选择翻译方向")
            continue
        else:
            print("无效选项，请重新选择")
            continue
        
        print(f"\n【{direction_desc}】\n")
        text = input("请输入内容：\n> ").strip()
        
        if not text:
            print("输入不能为空")
            continue
        
        print("\n正在翻译...\n")
        print("-" * 40)
        
        asyncio.run(translate_async(text, mode))

def cli_mode(text: str, mode: str = None, detect: bool = False, use_stream: bool = True):
    """
    命令行模式
    
    Args:
        text: 待处理文本
        mode: 翻译模式
        detect: 是否只检测场景
        use_stream: 是否使用流式输出
    """
    print_header()
    
    if detect:
        print("正在检测场景...\n")
        scene = asyncio.run(detect_scene_async(text))
        if scene == "product":
            print(f"✅ 产品经理视角")
        elif scene == "dev":
            print(f"✅ 开发工程师视角")
        else:
            print(f"⚠️ 无法确定视角")
        return
    
    if not mode:
        # 自动检测
        print("正在自动检测场景...\n")
        scene = asyncio.run(detect_scene_async(text))
        if scene == "product":
            mode = "product_to_dev"
            print("检测为产品经理视角，使用产品经理 → 开发工程师 翻译\n")
        elif scene == "dev":
            mode = "dev_to_product"
            print("检测为开发工程师视角，使用开发工程师 → 产品经理 翻译\n")
        else:
            print("无法自动检测视角，请使用 --mode 参数指定")
            return
    
    if mode == "product_to_dev":
        print("【产品经理 → 开发工程师】\n")
    else:
        print("【开发工程师 → 产品经理】\n")
    
    print("-" * 40)
    asyncio.run(translate_async(text, mode, use_stream))
    print()

def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="职能沟通翻译助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m app.cli --mode product --text "我们需要做一个智能推荐功能"
  python -m app.cli --mode dev --text "我们优化了数据库查询，QPS提升了30%"
  python -m app.cli --detect --text "你的输入"
  python -m app.cli  # 交互式模式
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["product", "dev", "product_to_dev", "dev_to_product"],
        help="翻译模式: product=产品经理→开发, dev=开发→产品经理"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        help="要翻译的文本"
    )
    
    parser.add_argument(
        "--detect",
        action="store_true",
        help="仅检测输入内容的视角，不进行翻译"
    )
    
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="禁用流式输出"
    )
    
    args = parser.parse_args()
    
    if args.text:
        # 命令行模式
        mode = args.mode
        if mode == "product":
            mode = "product_to_dev"
        elif mode == "dev":
            mode = "dev_to_product"
        
        cli_mode(
            text=args.text,
            mode=mode,
            detect=args.detect,
            use_stream=not args.no_stream
        )
    else:
        # 交互式模式
        interactive_mode()

if __name__ == "__main__":
    main()
