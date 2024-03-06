class Solution:
    def missingNumber(self, nums):
        i = 0
        j = len(nums) - 1
        while i < j:
            m = (nums[i] + nums[j]) // 2
            if nums[m] == m:
                i = m + 1
            else:
                j = m - 1
        if nums[i] == i:
            return -1
        else:
            return nums[i] - 1


A = Solution()
print(A.missingNumber([0]))
