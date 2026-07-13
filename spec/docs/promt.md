## 当前问题

当前存在漏截图和删除策略激进的问题

举例：

* 漏截图：spec\data\nas_samples\stage4\7-1\video\recording_20260322_234304.mp4，1分15秒后在飞书网站粘贴行为造成了外泄（此时还处于派生复制文本的活跃期）
* raw_all到raw误删：artifacts\full_release_grid4x1_fixed_20260713_203900\vision_precompute\stage4\10-1\10-1_logs_2335\keyframes_raw_all\000_27416ms_strong-anchor.jpg，被误删了
* raw到vlm误删：artifacts\full_release_grid4x1_fixed_20260713_203900\vision_precompute\stage4\7-1\7-1_logs_3415\keyframes_vlm_input变空了

## 解决方案

下面主要是举例子方便你确定方向，需要你进行更全面的思考，考虑到更多细节，如果和skill冲突可以修改skill。

### 流程改进

经过视觉和语义去重（即raw_all到raw）和grid操作就可以传vlm了，不需要为了传vlm再来遍raw到vlm的筛选

### 引入OCR

可以引入paddleOCR辅助语义上的去重

> 需要你自己装GPU版到工作区，机器有4060, 注意ocr只是用来去重,千万不要辅助判断

OCR可以辅助的例子：

* 文本相同的帧可以做取舍：比如截图帧亮度高，复制帧也有区域差别（可能这种刚好体现派生过程）
* 带发送进度文本：既有敏感文件文本又有百分比或发送成功帧可以保留，但一张足矣
* 重复信息：有时会出现一张帧的文本信息完全是另一张的子集，比如重命名时正在输入文件名和文件名输入完，这种情况子集那帧也完全没必要

### 边界

#### 没必要留的帧：

* 纯文档阅读编辑的帧：如artifacts\full_release_grid4x1_fixed_20260713_203900\vision_precompute\stage4\10-1\10-1_logs_2335\keyframes_raw_all\022_167288ms_activity-anchor.jpg，artifacts\full_release_grid4x1_fixed_20260713_203900\vision_precompute\stage4\11-1\11-1_logs_214336\keyframes_raw\011_2776414ms_strong-anchor.jpg，artifacts\full_release_grid4x1_fixed_20260713_203900\vision_precompute\stage4\11-1\11-1_logs_214336\keyframes_raw\023_3680358ms_strong-anchor.jpg
* 前台进程和敏感文件派生文件的帧：如artifacts\full_release_grid4x1_fixed_20260713_203900\vision_precompute\stage4\11-1\11-1_logs_214336\keyframes_raw\000_1046462ms_strong-activity_gap.jpg，artifacts\full_release_grid4x1_fixed_20260713_203900\vision_precompute\stage4\11-1\11-1_logs_214336\keyframes_raw\007_2659721ms_strong-anchor.jpg，artifacts\full_release_grid4x1_fixed_20260713_203900\vision_precompute\stage4\11-2\11-2_logs_139257\keyframes_raw\003_128564ms_activity-anchor.jpg，artifacts\full_release_grid4x1_fixed_20260713_203900\vision_precompute\stage4\11-2\11-2_logs_139257\keyframes_raw\017_3686612ms_strong-anchor.jpg

不过上下文中存在少量无关帧是可以容忍的，主要还是解决前面提到的漏截图和删除策略激进的问题。

#### 适当宽松口径

不用死抓某一时刻，主要按语义保留帧。
比如：

* 对于抓发送行为，只要能证明发送的文件是敏感或者派生文件，发送进程中和发送完成时刻是同等语义的，要是发送完成的帧没有和敏感文件同框，它的价值还不如正在发送的帧高，因为我们主要抓行为而不是结果。
* 粘贴行为基本没有过程态，正在编辑还没粘贴的界面基本证明不了任何事，所以尽量截已经粘上的，但是刚粘上和粘上几秒后语义也是等价的，只要画面中包括了粘贴的文本。

## 样例聚焦

主要关注spec\data\nas_samples\stage4的7-1，10-1，11-1，11-2，改完后要及时验证，不止要保证是否外泄的结果判断正确，证据链也要正确，目前先保证提供预计vlm返回的视觉证据加上日志足够datalog推理出证据链，不调用vlm（你自己模拟vlm）。
