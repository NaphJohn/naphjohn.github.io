## 数组
数组是存放在连续内存空间上的相同类型数据的集合。数组的元素是不能删的，只能覆盖。
需要两点注意的是
- 数组下标都是从0开始的。
- 数组内存空间的地址是连续的

移除元素，27(考虑双指针：https://leetcode.cn/problems/remove-element/；或者用栈的思想https://leetcode.cn/problems/remove-element/solutions/2802809/jian-dan-ti-jian-dan-zuo-pythonjavaccgoj-72bn/)

有序数组的平方，977(两边向中间合并：https://leetcode.cn/problems/squares-of-a-sorted-array/；https://leetcode.cn/problems/squares-of-a-sorted-array/solutions/2908673/python3javacgorustjavascript-yi-ti-yi-ji-v8im/)

长度最小的字数组，209，使用滑动窗口思想

螺旋矩阵，59

前缀和，58

## 链表

链表是一种通过指针串联在一起的线性结构，每一个节点由两部分组成，一个是数据域一个是指针域（存放指向下一个节点的指针），最后一个节点的指针域指向null（空指针的意思）。

**类型**

链表的入口节点称为链表的头结点也就是head。
如果遇到需要删除头节点的题目，添加哨兵节点可以简化代码逻辑，请记住这个技巧。
删除链表节点，基础的203

<details><summary>Details</summary>
<p>
主要思想：**递归**；

**迭代**（从前一个节点执行删除操作）
**迭代**

```python
初始化一个哨兵节点
cur = dummy = ListNode(-1, head)

或者写成
while head and head.val == val:
       head = head.next

```

</p>
</details>

反转链表，206
<details><summary>Details</summary>
<p>

使用双指针

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        cur, pre = head, None
        while cur:
            tmp = cur.next # 暂存后继节点 cur.next
            cur.next = pre # 修改 next 引用指向
            pre = cur      # pre 暂存 cur
            cur = tmp      # cur 访问下一节点
        return pre

```

还有递归思想
</p>
</details>

**删除倒数第N个节点**，19
使用哨兵节点
[https://leetcode.cn/problems/remove-nth-node-from-end-of-list/solutions/2004057/ru-he-shan-chu-jie-dian-liu-fen-zhong-ga-xpfs/]

[两两交换链表中的节点]，24

<details><summary>Details</summary>
<p>
算法:
创建哨兵节点 dummy,表示节点 0。
下面用 nodeo 表示 0,node1 表示 1,依此类推。
1.把nodeo 指向 node2。
2.把node2指向 nodel。
3.把node1指向 node3。
4.更新 nodeo为node1,更新 node1为node3。
5.如果node1和 node1.next 都不为空,
就回到第一步,执行下一或论交换。,
6.最后返回 dummy.next,作为新链表的头节点。

class Solution:
def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
node0 = dummy = ListNode(next=head)  # 用哨兵节点简化代码逻辑
node1 = head
while node1 and node1.next:  # 至少有两个节点
node2 = node1.next
node3 = node2.next

```
        node0.next = node2  # 0 -> 2
        node2.next = node1  # 2 -> 1
        node1.next = node3  # 1 -> 3

        node0 = node1  # 下一轮交换，0 是 1
        node1 = node3  # 下一轮交换，1 是 3
    return dummy.next  # 返回新链表的头节点

```

链接：https://leetcode.cn/problems/swap-nodes-in-pairs/solutions/2374872/tu-jie-die-dai-di-gui-yi-zhang-tu-miao-d-51ap/https://leetcode.cn/problems/swap-nodes-in-pairs/solutions/23876/bi-jiao-zhi-jie-gao-xiao-de-zuo-fa-han-tu-jie-by-w/
</p>
</details>

[面试题 02.07. 链表相交] (简单)
https://leetcode.cn/problems/intersection-of-two-linked-lists-lcci/solutions/1190240/mian-shi-ti-0207-lian-biao-xiang-jiao-sh-b8hn/

环形链表，142题
<details><summary>Details</summary>
<p>
[[https://leetcode.cn/problems/intersection-of-two-linked-lists-lcci/solutions/1190240/mian-shi-ti-0207-lian-biao-xiang-jiao-sh-b8hn/；](https://leetcode.cn/problems/intersection-of-two-linked-lists-lcci/solutions/1190240/mian-shi-ti-0207-lian-biao-xiang-jiao-sh-b8hn/%EF%BC%9B)](https://leetcode.cn/problems/intersection-of-two-linked-lists-lcci/solutions/1190240/mian-shi-ti-0207-lian-biao-xiang-jiao-sh-b8hn/%EF%BC%9B)https://leetcode.cn/problems/linked-list-cycle-ii/solutions/12616/linked-list-cycle-ii-kuai-man-zhi-zhen-shuang-zhi-/
</p>
</details> 

## 哈希表

一般哈希表都是用来快速判断一个元素是否出现集合里。

两个数组的交集，349，考虑使用一次遍历的方式，https://leetcode.cn/problems/intersection-of-two-arrays/solutions/3053420/yi-ci-bian-li-pythonjavacgojsrust-by-end-jstx/

快乐数，202

四数相加，454


## 字符串

最长回文子串，5

## 双指针法


## 栈与队列


## 二叉树

## 回溯算法


## 贪心算法


## 动态规划


## 数学